//! Port of `asmjit/aj_translate.cpp`: segment analysis and region formation.
//!
//! The execute segment is walked once with the decoder's own instruction
//! boundaries. Each instruction native code can run becomes an [`Insn`];
//! everything else stays with the interpreter. The compilable instructions
//! are then partitioned into non-overlapping *regions*, each compiled to one
//! native function: a region grows from an entry point along fall-through,
//! direct branches (so loops stay inside it) and over calls to their return
//! address, and stops at the entries of called functions, indirect jumps,
//! instructions it can't compile and instructions an earlier region already
//! claimed. Control moves between regions with native calls (a guest call
//! whose return address is in the caller's region, see [`super::calls`]) and tail
//! calls, never through the interpreter. The target of an `auipc`+`jalr`
//! pair is known, like a `jal`'s.
//!
//! The JIT and the ahead-of-time translator share this analysis, so both
//! run exactly the same instructions natively.

use std::collections::BTreeSet;

use super::classify::{JOp, classify, precise_rejects};
use crate::decoder_cache::{COMPRESSED, DecodedSegment};
pub(crate) use crate::native_code::block_icount;

/// The most instructions in one region. Regions are whole functions:
/// `rustc` (LLVM) takes superlinear time on large ones, and Cranelift's
/// register allocator slows down on the calls in them (2048 compiled
/// static glibc 15% slower than 512 and ran no faster).
const MAX_REGION_INSNS: usize = 512;
/// A region at its size limit still takes in the rest of a loop it holds
/// the head of, if the loop is at most this long (bytes, from the head to
/// the backward branch): cutting a loop in two would make every iteration
/// a native call into the other region. Up to this size in all.
const MAX_LOOP_BYTES: u64 = 1024;
const MAX_REGION_INSNS_WITH_LOOPS: usize = 2 * MAX_REGION_INSNS;

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
    /// `(insns index, target)` of every `jalr` whose target is a constant
    /// (see [`Analysis::jalr_target`]), ascending.
    jalr_targets: Vec<(usize, u64)>,
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

    /// The target of the `jalr` at `insns[i]` when it's a constant: the
    /// instruction before it sets its base register to a constant
    /// (`auipc`, `lui`, `li`) and nothing else reaches the `jalr` (it
    /// doesn't begin a block and isn't a branch target).
    pub(crate) fn jalr_target(&self, i: usize) -> Option<u64> {
        self.jalr_targets
            .binary_search_by_key(&i, |&(j, _)| j)
            .ok()
            .and_then(|k| self.jalr_targets.get(k))
            .map(|&(_, t)| t)
    }
}

/// The constant `jalr` targets of [`Analysis::jalr_target`].
fn find_jalr_targets(exec: &DecodedSegment, a: &Analysis) -> Vec<(usize, u64)> {
    let direct: BTreeSet<u64> = a
        .insns
        .iter()
        .filter_map(|ins| match ins.op {
            JOp::Branch { target, .. } | JOp::Jal { target, .. } => Some(target),
            _ => None,
        })
        .collect();
    let mut out = Vec::new();
    for (i, pair) in a.insns.windows(2).enumerate() {
        let [prev, ins] = pair else { continue };
        let (JOp::Const { rd, value }, JOp::Jalr { rs1, imm, .. }) = (prev.op, ins.op) else {
            continue;
        };
        if rd != rs1
            || rs1 == 0
            || prev.pc.wrapping_add(prev.len) != ins.pc
            || direct.contains(&ins.pc)
            || a.block_starts.binary_search(&exec.index(ins.pc)).is_ok()
        {
            continue;
        }
        out.push((i + 1, value.wrapping_add(imm as u64) & !1));
    }
    out
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
        jalr_targets: Vec::new(),
    };
    a.jalr_targets = find_jalr_targets(exec, &a);
    form_regions(exec, &mut a);
    a
}

/// True if `insns[i]` lies in one of the short `loops` whose head region
/// `id` holds.
fn in_held_loop(
    exec: &DecodedSegment,
    a: &Analysis,
    loops: &[(u64, u64)],
    id: RegionId,
    i: usize,
) -> bool {
    let Some(pc) = a.insns.get(i).map(|ins| ins.pc) else {
        return false;
    };
    // Heads within MAX_LOOP_BYTES before `pc`.
    let from = loops.partition_point(|&(h, _)| h < pc.saturating_sub(MAX_LOOP_BYTES));
    let to = loops.partition_point(|&(h, _)| h <= pc);
    loops
        .get(from..to)
        .unwrap_or_default()
        .iter()
        .any(|&(head, back)| pc <= back && a.region_at(exec, head) == Some(id))
}

/// The direct successors of `ins` that stay in its region: fall-through,
/// branch targets, `JAL x0` targets and a call's return address. Calls
/// and indirect jumps leave.
fn successors(ins: &Insn) -> [Option<u64>; 2] {
    let next = Some(ins.pc.wrapping_add(ins.len));
    match ins.op {
        JOp::Branch { target, .. } => [Some(target), next],
        JOp::Jal { rd: 0, target } => [Some(target), None],
        JOp::Jal { rd, .. } | JOp::Jalr { rd, .. } if rd != 0 => [next, None],
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
    // Function entries: what calls go to. A call's return address that is
    // one (after a call that doesn't return) begins another function, and
    // doesn't join the caller's region.
    let mut functions = BTreeSet::new();
    seeds.extend(a.jalr_targets.iter().map(|&(_, t)| t));
    for (i, ins) in a.insns.iter().enumerate() {
        match ins.op {
            JOp::Jal { rd, target } if rd != 0 => {
                functions.insert(target);
            }
            JOp::Jalr { rd, .. } if rd != 0 => {
                functions.extend(a.jalr_target(i));
            }
            _ => {}
        }
    }
    let fallback: Vec<u64> = a.insns.iter().map(|i| i.pc).collect();
    // Short loops, `(head, backward branch)`, by head.
    let mut loops: Vec<(u64, u64)> = a
        .insns
        .iter()
        .filter_map(|ins| match ins.op {
            JOp::Branch { target, .. } | JOp::Jal { rd: 0, target }
                if target <= ins.pc && ins.pc - target <= MAX_LOOP_BYTES =>
            {
                Some((target, ins.pc))
            }
            _ => None,
        })
        .collect();
    loops.sort_unstable();

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
            if region.insns.len() >= MAX_REGION_INSNS
                && (region.insns.len() >= MAX_REGION_INSNS_WITH_LOOPS
                    || !in_held_loop(exec, a, &loops, id, i))
            {
                // Left for another region.
                continue;
            }
            match a.region_of.get_mut(i) {
                Some(r) if *r == usize::MAX => *r = id,
                _ => continue,
            }
            region.insns.push(i);
            let Some(ins) = a.insns.get(i) else { continue };
            let call = matches!(ins.op, JOp::Jal { rd, .. } | JOp::Jalr { rd, .. } if rd != 0);
            for succ in successors(ins).into_iter().flatten() {
                if call && functions.contains(&succ) {
                    continue;
                }
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
        let jalr = a.jalr_target(i);
        let targets = match ins.op {
            JOp::Branch { target, .. } => [Some(target), Some(next)],
            JOp::Jal { target, .. } => [Some(target), None],
            JOp::Jalr { .. } => [jalr, None],
            _ => [Some(next), None],
        };
        for t in targets.into_iter().flatten() {
            if let Some(r) = a.region_at(exec, t) {
                let is_direct_target = jalr == Some(t)
                    || matches!(ins.op, JOp::Branch { target, .. } | JOp::Jal { target, .. } if target == t);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Machine, MachineOptions};

    const V: u64 = 0x10000;
    const ADDI: u32 = (1 << 20) | (10 << 15) | (10 << 7) | 0x13; // addi a0, a0, 1
    const DEC: u32 = (0xFFF << 20) | (11 << 15) | (11 << 7) | 0x13; // addi a1, a1, -1
    const JSELF: u32 = 0x6F; // j .

    /// `bnez a1, offset`
    const fn bnez(offset: i32) -> u32 {
        let o = offset as u32;
        ((o >> 12 & 1) << 31)
            | ((o >> 5 & 0x3F) << 25)
            | (11 << 15)
            | (1 << 12)
            | ((o >> 1 & 0xF) << 8)
            | ((o >> 11 & 1) << 7)
            | 0x63
    }

    fn analysed(words: &[u32]) -> (Machine, Analysis) {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut m = Machine::new(
            &[],
            MachineOptions {
                memory_max: 1 << 20,
                ..Default::default()
            },
        )
        .unwrap();
        m.init_execute_area(&code, V).unwrap();
        let a = analyse(m.cpu.current_execute_segment());
        (m, a)
    }

    /// A loop that the size limit would cut in two stays in one region:
    /// otherwise every iteration is a native call into the other one.
    #[test]
    fn the_size_limit_keeps_short_loops_whole() {
        let before = MAX_REGION_INSNS - 4;
        let mut words = vec![ADDI; before];
        // The loop: 8 instructions from its head to the branch back.
        words.extend([ADDI; 6]);
        words.push(DEC);
        words.push(bnez(-7 * 4));
        // After the loop: more than the limit allows.
        words.extend(vec![ADDI; MAX_REGION_INSNS]);
        words.push(JSELF);
        let (m, a) = analysed(&words);
        let exec = m.cpu.current_execute_segment();
        let region = |i: usize| a.region_at(exec, V + 4 * i as u64);
        let first = region(0);
        assert!(first.is_some());
        for i in before..before + 8 {
            assert_eq!(region(i), first, "loop instruction {i}");
        }
        // The code after the loop isn't: a region doesn't grow any further.
        assert_ne!(region(before + 8), first);
        assert!(
            a.regions
                .iter()
                .all(|r| r.insns.len() <= MAX_REGION_INSNS_WITH_LOOPS)
        );
    }

    /// Long loops are still cut at the limit.
    #[test]
    fn the_size_limit_cuts_long_loops() {
        let before = MAX_REGION_INSNS - 4;
        let len = (MAX_LOOP_BYTES / 4) as usize + 8;
        let mut words = vec![ADDI; before];
        words.extend(vec![ADDI; len - 2]);
        words.push(DEC);
        words.push(bnez(-((len as i32 - 1) * 4)));
        words.push(JSELF);
        let (m, a) = analysed(&words);
        let exec = m.cpu.current_execute_segment();
        let region = |i: usize| a.region_at(exec, V + 4 * i as u64);
        assert_eq!(region(MAX_REGION_INSNS - 1), region(0));
        assert_ne!(region(MAX_REGION_INSNS), region(0));
    }
}
