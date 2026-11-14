//! Port of the call analysis of `asmjit/aj_translate.cpp` and
//! `tr_emit.cpp` (`emit_function_call`): what native calls between
//! regions need to know. Both backends make a guest call (`jal`, or `jalr`
//! with a link register) a host call when its return address is a label of
//! the caller's region (regions keep it, see [`super::region`]). The caller stores the registers the callee may read,
//! calls, reloads those it may write and continues at the return address
//! if the callee returned there; any other return goes on to its own
//! caller.

use super::classify::{JOp, operands};
use super::region::{Analysis, RegionId};
use crate::decoder_cache::{COMPRESSED, DecodedSegment};

/// The most nested native calls; deeper calls are tail calls (which don't
/// grow the stack) and return through the entry table.
pub(crate) const DEPTH: u32 = 32;

/// What a region function returns instead of a guest return target: it
/// left for the host, with the exit in the state block. (Odd, so never a
/// `jalr` target.)
pub(crate) const DONE: u64 = 1;

/// Registers as a bit set: `x<n>` is bit `n`, `f<n>` bit `32 + n`.
pub(crate) type RegSet = u64;

/// What a region may read and write before it returns (a guest return,
/// or leaving for the host): its own registers and those of every region
/// it may transfer to, transitively; all of them if it may jump
/// indirectly. A native call to it stores the caller's registers it may
/// read, and reloads those it may write.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Summary {
    pub reads: RegSet,
    pub writes: RegSet,
}

/// The registers region `r` uses and writes (`x0` excluded). A
/// single-precision FP write keeps the upper half, so it reads it too.
pub(crate) fn region_regs(a: &Analysis, r: RegionId) -> Option<(RegSet, RegSet)> {
    let (mut used, mut written) = (0u64, 0u64);
    for &i in &a.regions.get(r)?.insns {
        let u = operands(a.insns.get(i)?.op);
        for s in u.x.into_iter().flatten().filter(|&s| s & 31 != 0) {
            used |= 1 << (s & 31);
        }
        if let Some(d) = u.xw.filter(|&d| d & 31 != 0) {
            used |= 1 << (d & 31);
            written |= 1 << (d & 31);
        }
        for s in u.f.into_iter().flatten() {
            used |= 1 << (32 + (s & 31));
        }
        if let Some(d) = u.fw {
            used |= 1 << (32 + (d & 31));
            written |= 1 << (32 + (d & 31));
        }
    }
    Some((used, written))
}

/// A guest return: `jalr x0, imm(ra)` or `jalr x0, imm(t0)`. It goes back
/// to the native caller (or `enter`) instead of through the entry table.
pub(crate) fn is_return(op: JOp) -> bool {
    matches!(op, JOp::Jalr { rd, rs1, .. } if rd & 31 == 0 && matches!(rs1 & 31, 1 | 5))
}

/// The constant target of the `jalr` at `a.insns[i]` that the code jumps
/// to directly, if any: one in a region (or else it goes through the
/// entry table), and aligned.
pub(crate) fn direct_jalr_target(exec: &DecodedSegment, a: &Analysis, i: usize) -> Option<u64> {
    a.jalr_target(i)
        .filter(|&t| COMPRESSED || t & 3 == 0)
        .filter(|&t| a.region_at(exec, t).is_some())
}

/// Every region's [`Summary`]: its own registers, then those of the
/// regions it may transfer to, to a fixed point.
pub(crate) fn summaries(exec: &DecodedSegment, a: &Analysis) -> Option<Vec<Summary>> {
    let n = a.regions.len();
    let mut out = Vec::with_capacity(n);
    let mut succ: Vec<Vec<RegionId>> = Vec::with_capacity(n);
    for (r, region) in a.regions.iter().enumerate() {
        let (used, written) = region_regs(a, r)?;
        let mut targets = std::collections::BTreeSet::new();
        let mut indirect = false;
        for &i in &region.insns {
            let ins = a.insns.get(i)?;
            let next = ins.pc.wrapping_add(ins.len);
            let mut to = |pc: u64| {
                if let Some(t) = a.region_at(exec, pc).filter(|&t| t != r) {
                    targets.insert(t);
                }
            };
            match ins.op {
                JOp::Branch { target, .. } => {
                    to(target);
                    to(next);
                }
                JOp::Jal { rd, target } => {
                    to(target);
                    if rd & 31 != 0 {
                        to(next);
                    }
                }
                JOp::Jalr { rd, .. } => {
                    match direct_jalr_target(exec, a, i) {
                        Some(t) => to(t),
                        None if is_return(ins.op) => {}
                        None => indirect = true,
                    }
                    if rd & 31 != 0 {
                        to(next);
                    }
                }
                _ => to(next),
            }
        }
        out.push(if indirect {
            Summary {
                reads: !0,
                writes: !0,
            }
        } else {
            Summary {
                reads: used,
                writes: written,
            }
        });
        succ.push(targets.into_iter().collect());
    }
    loop {
        let mut changed = false;
        for (r, targets) in succ.iter().enumerate() {
            let mut s = *out.get(r)?;
            for &t in targets {
                let o = *out.get(t)?;
                s.reads |= o.reads;
                s.writes |= o.writes;
            }
            let slot = out.get_mut(r)?;
            if *slot != s {
                *slot = s;
                changed = true;
            }
        }
        if !changed {
            return Some(out);
        }
    }
}
