//! Port of `tr_emit.cpp` (the binary translator's emitter, which writes C),
//! writing safe Rust instead, for the regions of one execute segment. The
//! instruction selection, region formation and counting are the JIT's
//! (`codegen`, `jit/emit.rs`), so translated code runs exactly the
//! instructions the JIT compiles, with the same results and counts.
//!
//! Each segment becomes a module, each region a function
//!
//! ```text
//! extern "rust-preserve-none" fn r<N>(s: &mut S<'_>, pc: u64, c: u64)
//! ```
//!
//! that keeps the guest registers it uses in locals (C++ `use_register_caching`),
//! dispatches on `pc` in a `loop { match pc { .. } }` whose arms are the
//! region's entry points (the counterpart of C++'s `_jumptbl` switch: LLVM
//! threads the constant `pc` of direct transfers into plain branches), and
//! leaves through one shared exit, which stores the registers the region
//! wrote and then either returns to the host with an `EXIT_*` kind, or
//! tail-calls the next region with `become`: directly for direct transfers,
//! through the segment's entry table for indirect jumps. C++'s translated
//! functions can't call each other that way (C has no guaranteed tail
//! calls) and return to the dispatcher instead.
//!
//! # Memory safety
//!
//! The generated code is safe Rust (`forbid(unsafe_code)`): it can't touch
//! anything but the registers, the arena and its own state, whatever it
//! does. The arena is a `&mut [u8; N]`, where `N` covers the windows of the
//! layout the code is built for plus the widest access, so after the
//! window check every index is provably in bounds and the compiler emits
//! no second check: one compare per access, as in C++. An access outside
//! the windows leaves at that instruction (`EXIT_SLOW`), like the JIT's.
//! With an N-bit address space the address is masked instead, and there is
//! no check at all.

use std::fmt::Write as _;

use super::structure::{dispatch, structure};
use crate::codegen::classify::{AluI, AluR, Cond, FBin, FCmp, FFma, FInt, JOp, Unary, operands};
use crate::codegen::region::{Analysis, Insn, RegionId, block_icount};
use crate::decoder_cache::{COMPRESSED, DecodedSegment, SHIFT};
use crate::native_code::{
    EXIT_DYNCALL, EXIT_HALT, EXIT_INDIRECT, EXIT_JUMP, EXIT_MISALIGNED, EXIT_SLOW, MemoryLayout,
};

/// The widest guest memory access, in bytes (FLD/FSD/LD/SD).
const MAX_ACCESS: u64 = 8;

/// One translated segment: its module's source, and what the host needs to
/// enter it.
#[derive(Debug, Clone)]
pub(crate) struct Translation {
    /// The source of the segment's module (`mod s<N> { .. }` without the
    /// header line, which the writer adds with the module number).
    pub source: String,
    /// The number of decoder entries of the segment.
    pub cache_len: usize,
    /// Decoder indices with an entry into native code, ascending.
    pub entries: Vec<u32>,
    /// The decoder entries to patch: block starts with an entry, ascending.
    pub patch: Vec<u32>,
}

/// The code shared by all segments of a translation: the exit enum and the
/// pure helpers, all `#[inline(always)]`. Matches the interpreter's (and
/// the JIT's) results for every input.
pub(crate) const RUNTIME: &str = r#"
    /// How a region function leaves: an exit to the host (kind, PC), a
    /// direct tail call (region, PC) or an indirect jump (target).
    pub(super) enum A {
        X(u64, u64),
        G(u32, u64),
        I(u64),
    }
    #[inline(always)]
    pub(super) fn sx(v: u32) -> u64 {
        v as i32 as i64 as u64
    }
    #[inline(always)]
    pub(super) fn mulh(a: u64, b: u64) -> u64 {
        ((i128::from(a as i64) * i128::from(b as i64)) >> 64) as u64
    }
    #[inline(always)]
    pub(super) fn mulhsu(a: u64, b: u64) -> u64 {
        ((i128::from(a as i64) * i128::from(b)) >> 64) as u64
    }
    #[inline(always)]
    pub(super) fn mulhu(a: u64, b: u64) -> u64 {
        ((u128::from(a) * u128::from(b)) >> 64) as u64
    }
    #[inline(always)]
    pub(super) fn div(a: u64, b: u64) -> u64 {
        let (a, b) = (a as i64, b as i64);
        if b == 0 { u64::MAX } else if b == -1 { a.wrapping_neg() as u64 } else { (a / b) as u64 }
    }
    #[inline(always)]
    pub(super) fn rem(a: u64, b: u64) -> u64 {
        let (a, b) = (a as i64, b as i64);
        if b == 0 { a as u64 } else if b == -1 { 0 } else { (a % b) as u64 }
    }
    #[inline(always)]
    pub(super) fn divu(a: u64, b: u64) -> u64 {
        if b == 0 { u64::MAX } else { a / b }
    }
    #[inline(always)]
    pub(super) fn remu(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { a % b }
    }
    #[inline(always)]
    pub(super) fn divw(a: u64, b: u64) -> u64 {
        let (a, b) = (a as i32, b as i32);
        sx(if b == 0 { u32::MAX } else if b == -1 { a.wrapping_neg() as u32 } else { (a / b) as u32 })
    }
    #[inline(always)]
    pub(super) fn remw(a: u64, b: u64) -> u64 {
        let (a, b) = (a as i32, b as i32);
        sx(if b == 0 { a as u32 } else if b == -1 { 0 } else { (a % b) as u32 })
    }
    #[inline(always)]
    pub(super) fn divuw(a: u64, b: u64) -> u64 {
        let (a, b) = (a as u32, b as u32);
        sx(if b == 0 { u32::MAX } else { a / b })
    }
    #[inline(always)]
    pub(super) fn remuw(a: u64, b: u64) -> u64 {
        let (a, b) = (a as u32, b as u32);
        sx(if b == 0 { a } else { a % b })
    }
    #[inline(always)]
    pub(super) fn orcb(a: u64) -> u64 {
        const LOW7: u64 = 0x7F7F_7F7F_7F7F_7F7F;
        ((((a & LOW7).wrapping_add(LOW7) | a) & 0x8080_8080_8080_8080) >> 7).wrapping_mul(0xFF)
    }
    #[inline(always)]
    pub(super) fn fs(b: u64) -> f32 {
        f32::from_bits(b as u32)
    }
    #[inline(always)]
    pub(super) fn fd(b: u64) -> f64 {
        f64::from_bits(b)
    }
    /// FCVT.S.D: a NaN gives the canonical NaN.
    #[inline(always)]
    pub(super) fn cvt_s_d(x: f64) -> f32 {
        if x.is_nan() { f32::from_bits(0x7FC0_0000) } else { x as f32 }
    }
    /// FCVT.D.S: a NaN gives the canonical NaN.
    #[inline(always)]
    pub(super) fn cvt_d_s(x: f32) -> f64 {
        if x.is_nan() { f64::from_bits(0x7FF8_0000_0000_0000) } else { f64::from(x) }
    }
    /// A single-precision result in the low half, keeping the upper one.
    #[inline(always)]
    pub(super) fn ss(old: u64, v: f32) -> u64 {
        (old & 0xFFFF_FFFF_0000_0000) | u64::from(v.to_bits())
    }
    #[inline(always)]
    pub(super) fn fmin_s(a: f32, b: f32) -> f32 {
        if a.is_nan() && b.is_nan() { return f32::from_bits(0x7FC0_0000); }
        if a == 0.0 && b == 0.0 { return f32::from_bits((a.to_bits() | b.to_bits()) & 0x8000_0000); }
        a.min(b)
    }
    #[inline(always)]
    pub(super) fn fmax_s(a: f32, b: f32) -> f32 {
        if a.is_nan() && b.is_nan() { return f32::from_bits(0x7FC0_0000); }
        if a == 0.0 && b == 0.0 { return f32::from_bits(a.to_bits() & b.to_bits() & 0x8000_0000); }
        a.max(b)
    }
    #[inline(always)]
    pub(super) fn fmin_d(a: f64, b: f64) -> f64 {
        if a.is_nan() && b.is_nan() { return f64::from_bits(0x7FF8_0000_0000_0000); }
        if a == 0.0 && b == 0.0 { return f64::from_bits((a.to_bits() | b.to_bits()) & (1 << 63)); }
        a.min(b)
    }
    #[inline(always)]
    pub(super) fn fmax_d(a: f64, b: f64) -> f64 {
        if a.is_nan() && b.is_nan() { return f64::from_bits(0x7FF8_0000_0000_0000); }
        if a == 0.0 && b == 0.0 { return f64::from_bits(a.to_bits() & b.to_bits() & (1 << 63)); }
        a.max(b)
    }
    /// `fcvt_round`: the RISC-V rounding mode `rm` (reserved modes, and
    /// DYN's FCSR.frm values beyond 4, round to nearest-even).
    #[inline(always)]
    pub(super) fn round_s(x: f32, rm: u64) -> f32 {
        match rm { 1 => x.trunc(), 2 => x.floor(), 3 => x.ceil(), 4 => x.round(), _ => x.round_ties_even() }
    }
    #[inline(always)]
    pub(super) fn round_d(x: f64, rm: u64) -> f64 {
        match rm { 1 => x.trunc(), 2 => x.floor(), 3 => x.ceil(), 4 => x.round(), _ => x.round_ties_even() }
    }
    // FCVT to integers: round, convert with saturation, NaN to the maximum;
    // 32-bit results sign-extended.
    #[inline(always)]
    pub(super) fn cvt_w_s(x: f32, rm: u64) -> u64 {
        if x.is_nan() { i32::MAX as u64 } else { round_s(x, rm) as i32 as i64 as u64 }
    }
    #[inline(always)]
    pub(super) fn cvt_wu_s(x: f32, rm: u64) -> u64 {
        if x.is_nan() { u64::MAX } else { sx(round_s(x, rm) as u32) }
    }
    #[inline(always)]
    pub(super) fn cvt_l_s(x: f32, rm: u64) -> u64 {
        if x.is_nan() { i64::MAX as u64 } else { round_s(x, rm) as i64 as u64 }
    }
    #[inline(always)]
    pub(super) fn cvt_lu_s(x: f32, rm: u64) -> u64 {
        if x.is_nan() { u64::MAX } else { round_s(x, rm) as u64 }
    }
    #[inline(always)]
    pub(super) fn cvt_w_d(x: f64, rm: u64) -> u64 {
        if x.is_nan() { i32::MAX as u64 } else { round_d(x, rm) as i32 as i64 as u64 }
    }
    #[inline(always)]
    pub(super) fn cvt_wu_d(x: f64, rm: u64) -> u64 {
        if x.is_nan() { u64::MAX } else { sx(round_d(x, rm) as u32) }
    }
    #[inline(always)]
    pub(super) fn cvt_l_d(x: f64, rm: u64) -> u64 {
        if x.is_nan() { i64::MAX as u64 } else { round_d(x, rm) as i64 as u64 }
    }
    #[inline(always)]
    pub(super) fn cvt_lu_d(x: f64, rm: u64) -> u64 {
        if x.is_nan() { u64::MAX } else { round_d(x, rm) as u64 }
    }
    /// `rv_fclass32`/`rv_fclass64` on the raw bits: exactly one bit set.
    #[inline(always)]
    pub(super) fn fclass(bits: u64, sign_bit: u32, exp_shift: u32, exp_max: u64, frac_mask: u64, quiet: u64) -> u64 {
        let negative = (bits >> sign_bit) & 1 != 0;
        let exp = (bits >> exp_shift) & exp_max;
        let frac = bits & frac_mask;
        let bit = if exp == exp_max {
            if frac == 0 { if negative { 0 } else { 7 } } else if frac & quiet != 0 { 9 } else { 8 }
        } else if exp == 0 {
            if frac == 0 { if negative { 3 } else { 4 } } else if negative { 2 } else { 5 }
        } else if negative { 1 } else { 6 };
        1 << bit
    }
"#;

/// Translate the regions of `a` into the source of `exec`'s module, for
/// guest memory `layout`. `None` if there is nothing to translate.
pub(crate) fn translate(
    exec: &DecodedSegment,
    a: &Analysis,
    layout: MemoryLayout,
) -> Option<Translation> {
    if a.regions.is_empty() {
        return None;
    }
    let cache_len = exec.cache.len();
    let mut entries: Vec<(u32, RegionId)> = Vec::new();
    for (r, region) in a.regions.iter().enumerate() {
        for &pc in &region.entries {
            let i = exec.index(pc);
            if i < cache_len {
                entries.push((u32::try_from(i).ok()?, r));
            }
        }
    }
    entries.sort_unstable();
    entries.dedup_by_key(|e| e.0);
    let is_entry = |i: usize| {
        entries
            .binary_search_by_key(&(i as u64), |&(e, _)| u64::from(e))
            .is_ok()
    };
    let patch: Vec<u32> = a
        .block_starts
        .iter()
        .copied()
        .filter(|&i| is_entry(i))
        .map(u32::try_from)
        .collect::<Result<_, _>>()
        .ok()?;

    let mut out = String::new();
    let o = &mut out;
    let begin = exec.exec_begin();
    // The arena prefix the code may touch: the windows (or the mask) plus
    // the widest access. The host passes the whole arena, which is at least
    // that long whenever the layout is the machine's.
    let n = if layout.mask != 0 {
        layout.mask.checked_add(MAX_ACCESS)?
    } else {
        let end = |(base, len): (u64, u64)| base.checked_add(len);
        end(layout.read)?
            .max(end(layout.write)?)
            .checked_add(MAX_ACCESS - 1)?
    };
    let _ = writeln!(o, "        use super::rt::*;");
    let _ = writeln!(o, "        const BEGIN: u64 = {begin:#x};");
    let _ = writeln!(o, "        const N: usize = {n:#x};");
    let _ = writeln!(o, "        const CACHE_LEN: usize = {cache_len};");
    let (mask, (rb, rl), (wb, wl)) = (layout.mask, layout.read, layout.write);
    if mask != 0 {
        let _ = writeln!(
            o,
            r#"        const MASK: u64 = {mask:#x};
        #[inline(always)]
        fn ld<const W: usize>(m: &[u8; N], a: u64) -> Option<[u8; W]> {{
            let i = (a & MASK) as usize;
            m.get(i..i + W)?.try_into().ok()
        }}
        #[inline(always)]
        fn st<const W: usize>(m: &mut [u8; N], a: u64, v: [u8; W]) -> bool {{
            let i = (a & MASK) as usize;
            match m.get_mut(i..i + W) {{
                Some(b) => {{ b.copy_from_slice(&v); true }}
                None => false,
            }}
        }}"#
        );
    } else {
        let _ = writeln!(
            o,
            r#"        // The arena windows, as `Memory::read_*`/`write_*` check them:
        // only the start address. N covers the last byte of the widest
        // access at the last address of either window.
        const RB: u64 = {rb:#x};
        const RL: u64 = {rl:#x};
        const WB: u64 = {wb:#x};
        const WL: u64 = {wl:#x};
        #[inline(always)]
        fn ld<const W: usize>(m: &[u8; N], a: u64) -> Option<[u8; W]> {{
            if a.wrapping_sub(RB) >= RL {{ return None; }}
            let i = a as usize;
            m.get(i..i + W)?.try_into().ok()
        }}
        #[inline(always)]
        fn st<const W: usize>(m: &mut [u8; N], a: u64, v: [u8; W]) -> bool {{
            if a.wrapping_sub(WB) >= WL {{ return false; }}
            let i = a as usize;
            match m.get_mut(i..i + W) {{
                Some(b) => {{ b.copy_from_slice(&v); true }}
                None => false,
            }}
        }}
        // Accesses covered by a span check: `k` was compared with a
        // window's length minus the span, `d` is the window's base plus
        // the access's offset in the span, so `k + d + W <= N`.
        #[inline(always)]
        fn ldk<const W: usize>(m: &[u8; N], k: u64, d: usize) -> Option<[u8; W]> {{
            let i = (k as usize).wrapping_add(d);
            m.get(i..i.wrapping_add(W))?.try_into().ok()
        }}
        #[inline(always)]
        fn stk<const W: usize>(m: &mut [u8; N], k: u64, d: usize, v: [u8; W]) -> bool {{
            let i = (k as usize).wrapping_add(d);
            match m.get_mut(i..i.wrapping_add(W)) {{
                Some(b) => {{ b.copy_from_slice(&v); true }}
                None => false,
            }}
        }}"#
        );
    }
    let _ = writeln!(
        o,
        r#"        pub(super) struct S<'a> {{
            x: &'a mut [u64; 32],
            f: &'a mut [u64; 32],
            m: &'a mut [u8; N],
            max: u64,
            frm: u64,
            t: &'static [Option<F>],
            out: [u64; 3],
        }}
        type F = for<'a, 'b> extern "rust-preserve-none" fn(&'a mut S<'b>, u64, u64);
        static TABLE: std::sync::OnceLock<Box<[Option<F>]>> = std::sync::OnceLock::new();
        /// The entry table, by decoder index, for indirect jumps.
        fn table() -> Box<[Option<F>]> {{
            let mut t = vec![None; CACHE_LEN].into_boxed_slice();
            for &(i, f) in ENTRIES {{
                if let Some(e) = t.get_mut(i as usize) {{
                    *e = Some(f);
                }}
            }}
            t
        }}
        pub(super) fn enter(index: usize, x: &mut [u64; 32], f: &mut [u64; 32], arena: &mut [u8], io: &mut [u64; 4]) -> bool {{
            let Some(m) = arena.get_mut(..N) else {{ return false }};
            let Ok(m) = <&mut [u8; N]>::try_from(m) else {{ return false }};
            let t: &'static [Option<F>] = TABLE.get_or_init(table);
            let Some(&Some(func)) = t.get(index) else {{ return false }};
            let [pc, counter, max, frm] = *io;
            let mut s = S {{ x, f, m, max, frm, t, out: [u64::MAX, pc, counter] }};
            func(&mut s, pc, counter);
            let [kind, pc, counter] = s.out;
            *io = [kind, pc, counter, 0];
            true
        }}"#
    );
    let _ = writeln!(o, "        static ENTRIES: &[(u32, F)] = &[");
    for &(i, r) in &entries {
        let _ = writeln!(o, "            ({i}, r{r} as F),");
    }
    let _ = writeln!(o, "        ];");
    for r in 0..a.regions.len() {
        RegionWriter::write(exec, a, r, layout, o)?;
    }
    Some(Translation {
        source: out,
        cache_len,
        entries: entries.iter().map(|&(i, _)| i).collect(),
        patch,
    })
}

/// A piece of a region's code: Rust source, or a transfer to one of the
/// region's labels, which is only rendered once the region's control flow
/// is laid out (see `structure`).
#[derive(Debug, Clone)]
pub(super) enum Item {
    Line(String),
    Branch(u64),
}

/// The code of one label (entry point): from the label to the next label,
/// or to the instruction that ends it.
#[derive(Debug)]
pub(super) struct Arm {
    pub pc: u64,
    pub items: Vec<Item>,
}

/// Writes one region function.
struct RegionWriter<'a> {
    exec: &'a DecodedSegment,
    a: &'a Analysis,
    region: RegionId,
    /// The region's entry points: its labels.
    labels: &'a std::collections::BTreeSet<u64>,
    /// Regions this one tail-calls directly.
    calls: std::collections::BTreeSet<RegionId>,
    arms: Vec<Arm>,
    /// The arm being written.
    items: Vec<Item>,
    layout: MemoryLayout,
    /// Position in the region's instruction list of the instruction being
    /// written.
    pos: usize,
    /// `[load, store][base register]`: the span of offsets an earlier check
    /// in this straight-line run covers for accesses through that register,
    /// the window it checked, and the local holding `base + lo - window`.
    checked: [[Option<Check>; 32]; 2],
    /// Checks written so far (names their locals).
    checks: usize,
    /// `[load, store][base register]`: the check made on entry to the
    /// region for a base register the region never writes (see
    /// [`RegionWriter::region_checks`]). A run's first access through the
    /// register only tests its result.
    region_checked: [[Option<Check>; 32]; 2],
    /// The entry code of the region checks.
    region_lines: Vec<String>,
}

/// Offsets `lo..=hi` of guest memory accesses through one base register.
type Span = (i64, i64);

/// A check covering the accesses through one base register (see
/// [`RegionWriter::access`]).
#[derive(Debug, Clone, Copy)]
struct Check {
    lo: i64,
    hi: i64,
    /// The window's base.
    base: u64,
    /// The number of the local `k<n>`.
    local: usize,
}

/// A Rust expression for integer register `r` (`x0` is the constant 0).
fn x(r: u32) -> String {
    match r & 31 {
        0 => "0u64".to_owned(),
        r => format!("x{r}"),
    }
}

/// A Rust expression for the raw bits of FP register `r`.
fn f(r: u32) -> String {
    format!("f{}", r & 31)
}

/// `imm` as a `u64` literal (two's complement).
fn lit(imm: i64) -> String {
    format!("{:#x}u64", imm as u64)
}

impl<'a> RegionWriter<'a> {
    fn write(
        exec: &'a DecodedSegment,
        a: &'a Analysis,
        region: RegionId,
        layout: MemoryLayout,
        o: &mut String,
    ) -> Option<()> {
        let r = a.regions.get(region)?;
        let (mut used, mut written) = ([false; 32], [false; 32]);
        let (mut fused, mut fwritten) = ([false; 32], [false; 32]);
        for &i in &r.insns {
            let u = operands(a.insns.get(i)?.op);
            for s in u.x.into_iter().flatten().filter(|&s| s != 0) {
                *used.get_mut(s as usize)? = true;
            }
            if let Some(d) = u.xw.filter(|&d| d != 0) {
                *used.get_mut(d as usize)? = true;
                *written.get_mut(d as usize)? = true;
            }
            for s in u.f.into_iter().flatten() {
                *fused.get_mut(s as usize)? = true;
            }
            if let Some(d) = u.fw {
                // A single-precision write keeps the upper half: read it too.
                *fused.get_mut(d as usize)? = true;
                *fwritten.get_mut(d as usize)? = true;
            }
        }
        let mut w = RegionWriter {
            exec,
            a,
            region,
            labels: &r.entries,
            calls: std::collections::BTreeSet::new(),
            arms: Vec::new(),
            items: Vec::new(),
            layout,
            pos: 0,
            checked: [[None; 32]; 2],
            checks: 0,
            region_checked: [[None; 32]; 2],
            region_lines: Vec::new(),
        };
        w.region_checks(&r.insns, &written)?;
        w.body(&r.insns)?;
        // Without the limit nothing is counted (`MemoryLayout::unlimited`).
        let entry_counts: Vec<(u64, u64)> = r
            .entries
            .iter()
            .map(|&pc| {
                (
                    pc,
                    if layout.unlimited {
                        0
                    } else {
                        block_icount(exec, pc)
                    },
                )
            })
            .collect();
        let control =
            structure(&w.arms, &entry_counts).unwrap_or_else(|| dispatch(&w.arms, &entry_counts));

        let _ = writeln!(
            o,
            "        extern \"rust-preserve-none\" fn r{region}(s: &mut S<'_>, pc: u64, c: u64) {{"
        );
        for n in 1..32 {
            if used.get(n).copied().unwrap_or(false) {
                let _ = writeln!(o, "            let mut x{n} = s.x[{n}];");
            }
        }
        for n in 0..32 {
            if fused.get(n).copied().unwrap_or(false) {
                let _ = writeln!(o, "            let mut f{n} = s.f[{n}];");
            }
        }
        // The arena, the limit and the rounding mode in locals: read
        // through `s`, they would be reloaded after every guest store
        // (`s` escapes into the tail calls, so LLVM can't tell that the
        // arena doesn't overlap it).
        let _ = writeln!(o, "            let m = &mut *s.m;");
        let _ = writeln!(o, "            let (max, frm) = (s.max, s.frm);");
        for line in &w.region_lines {
            let _ = writeln!(o, "            {line}");
        }
        let _ = writeln!(o, "            let mut c = c;");
        let _ = writeln!(o, "            let a: A = 'x: {{");
        let _ = writeln!(o, "                let mut pc = pc;");
        o.push_str(&control);
        let _ = writeln!(o, "            }};");
        for n in 1..32 {
            if written.get(n).copied().unwrap_or(false) {
                let _ = writeln!(o, "            s.x[{n}] = x{n};");
            }
        }
        for n in 0..32 {
            if fwritten.get(n).copied().unwrap_or(false) {
                let _ = writeln!(o, "            s.f[{n}] = f{n};");
            }
        }
        let _ = writeln!(o, "            match a {{");
        let _ = writeln!(o, "                A::X(k, p) => s.out = [k, p, c],");
        let _ = writeln!(o, "                A::G(r, p) => match r {{");
        for &callee in &w.calls {
            let _ = writeln!(
                o,
                "                    {callee} => become r{callee}(s, p, c),"
            );
        }
        let _ = writeln!(o, "                    _ => s.out = [{EXIT_JUMP}, p, c],");
        let _ = writeln!(o, "                }},");
        let _ = writeln!(
            o,
            "                A::I(t) => {{
                    let i = ((t >> {SHIFT}).wrapping_sub(BEGIN >> {SHIFT})) as usize;
                    if let Some(f) = s.t.get(i).copied().flatten() {{
                        become f(s, t, c)
                    }}
                    s.out = [{EXIT_INDIRECT}, t, c];
                }}"
        );
        let _ = writeln!(o, "            }}");
        let _ = writeln!(o, "        }}");
        Some(())
    }

    fn line(&mut self, code: &str) {
        self.items.push(Item::Line(code.to_owned()));
    }

    /// Transfer to `target`, a label of this region.
    fn branch(&mut self, target: u64) {
        self.items.push(Item::Branch(target));
    }

    /// End the arm being written.
    fn finish_arm(&mut self, pc: u64) {
        let items = std::mem::take(&mut self.items);
        self.arms.push(Arm { pc, items });
    }

    fn body(&mut self, insns: &[usize]) -> Option<()> {
        // `open`: the current arm can still fall through.
        let mut open = false;
        let mut arm_pc = 0;
        for (pos, &i) in insns.iter().enumerate() {
            let ins = *self.a.insns.get(i)?;
            if self.labels.contains(&ins.pc) {
                if open {
                    // Sequential flow into an entry point: already counted.
                    self.branch(ins.pc);
                    self.finish_arm(arm_pc);
                }
                arm_pc = ins.pc;
                // Other paths reach the label without this run's checks.
                self.checked = [[None; 32]; 2];
            } else if !open {
                // Unreachable by construction: a non-entry instruction is
                // only ever reached by falling through.
                return None;
            }
            self.pos = pos;
            open = self.instruction(&ins)?;
            // Checks made through the old value of a written register no
            // longer apply.
            if let Some(rd) = operands(ins.op).xw
                && let Some(r) = usize::try_from(rd & 31).ok()
            {
                for kind in &mut self.checked {
                    if let Some(c) = kind.get_mut(r) {
                        *c = None;
                    }
                }
            }
            let next = ins.pc.wrapping_add(ins.len);
            if open && ins.funcblock {
                // C++ `RV32I_BC_FUNCBLOCK`: `NEXT_BLOCK` with the overflow
                // check.
                self.check_counter(next);
                self.enter_block(next);
                open = false;
            } else if open && self.a.region_at(self.exec, next) != Some(self.region) {
                self.leave_mid_block(next);
                open = false;
            }
            if !open {
                self.finish_arm(arm_pc);
            }
        }
        if open {
            return None;
        }
        Some(())
    }

    /// Continue at `target` as the start of a block (C++ `BEGIN_BLOCK`).
    fn enter_block(&mut self, target: u64) {
        match self.a.region_at(self.exec, target) {
            Some(r) if r == self.region && self.labels.contains(&target) => {
                if !self.layout.unlimited {
                    let k = block_icount(self.exec, target);
                    self.line(&format!("c = c.wrapping_add({k});"));
                }
                self.branch(target);
            }
            // Every in-region target is an entry; be safe anyway.
            Some(r) if r == self.region => self.leave(EXIT_JUMP, target),
            Some(r) => {
                self.calls.insert(r);
                self.line(&format!("break 'x A::G({r}, {target:#x});"));
            }
            None => self.leave(EXIT_JUMP, target),
        }
    }

    /// Continue at `target` in the middle of a block: whoever runs it adds
    /// the block count for `target` again, so subtract it first.
    fn leave_mid_block(&mut self, target: u64) {
        if !self.layout.unlimited {
            let k = block_icount(self.exec, target);
            self.line(&format!("c = c.wrapping_sub({k});"));
        }
        match self.a.region_at(self.exec, target) {
            Some(r) => {
                self.calls.insert(r);
                self.line(&format!("break 'x A::G({r}, {target:#x});"));
            }
            None => self.leave(EXIT_JUMP, target),
        }
    }

    fn leave(&mut self, kind: u64, pc: u64) {
        self.line(&format!("break 'x A::X({kind}, {pc:#x});"));
    }

    /// `if counter >= max { halt at target }`, as the dispatcher checks
    /// before jumping. Nothing without the limit.
    fn check_counter(&mut self, target: u64) {
        if self.layout.unlimited {
            return;
        }
        self.line(&format!(
            "if c >= max {{ break 'x A::X({EXIT_HALT}, {target:#x}); }}"
        ));
    }

    /// The guest address `rs1 + imm`.
    fn addr(rs1: u32, imm: i64) -> String {
        if rs1 & 31 == 0 {
            lit(imm)
        } else {
            format!("{}.wrapping_add({})", x(rs1), lit(imm))
        }
    }

    /// A guest memory access of `width` bytes at `rs1 + imm` by the
    /// instruction at `pc`: an expression for `ld::<width>`/`st::<width>`'s
    /// first arguments (`ldk`/`stk` with a checked local, in a checked
    /// layout), after writing the check it needs, if any. Like `Memory`'s
    /// fast path, an access outside the windows goes to the page table:
    /// the interpreter takes over at `pc`.
    ///
    /// As in the JIT, the first access through a base register checks the
    /// span of offsets of the accesses through it up to the end of its
    /// straight-line run (or until the register changes), and the others
    /// have no check of their own: the check computes `k = rs1 + lo -
    /// window base` and compares it with the window's length minus the
    /// span, so `k + (offset - lo)` is below the length for every access
    /// in the span, which the compiler can prove from the comparison. The
    /// interpreter checks each access's start address, so the span check
    /// passes exactly when all of theirs would; when it fails, native code
    /// leaves at the first access and the interpreter runs the rest of the
    /// block, as it would for that access alone.
    fn access(&mut self, rs1: u32, imm: i64, write: bool, pc: u64) -> (&'static str, String) {
        if self.layout.mask != 0 {
            return ("", format!("m, {}", Self::addr(rs1, imm)));
        }
        let reg = (rs1 & 31) as usize;
        let within = |c: Option<Check>| c.filter(|c| c.lo <= imm && imm <= c.hi);
        let check = |kind: usize, me: &Self| {
            me.checked
                .get(kind)
                .and_then(|k| k.get(reg))
                .copied()
                .flatten()
        };
        // A store check also covers loads when every writable address is
        // readable.
        let stores_cover_loads = window_within(self.layout.write, self.layout.read);
        let covered = if write {
            within(check(1, self))
        } else {
            within(check(0, self)).or_else(|| within(check(1, self)).filter(|_| stores_cover_loads))
        };
        let region = self
            .region_checked
            .get(usize::from(write))
            .and_then(|k| k.get(reg))
            .copied()
            .flatten();
        let c = match (covered, region) {
            (Some(c), _) => c,
            (None, Some(c)) => {
                self.line(&format!(
                    "if !o{} {{ break 'x A::X({EXIT_SLOW}, {pc:#x}); }}",
                    c.local
                ));
                if let Some(slot) = self
                    .checked
                    .get_mut(usize::from(write))
                    .and_then(|k| k.get_mut(reg))
                {
                    *slot = Some(c);
                }
                c
            }
            (None, None) => {
                let (loads, stores) = self.check_spans(rs1);
                let inside =
                    |span: Option<(i64, i64)>| span.is_some_and(|(lo, hi)| lo <= imm && imm <= hi);
                // A load inside the span of the stores that follow it is
                // checked with them: it lies between two of their
                // addresses, so it is inside the window when they are.
                let as_store = write || (stores_cover_loads && inside(stores));
                let (lo, hi) = if as_store { stores } else { loads }.unwrap_or((imm, imm));
                let (base, len) = if as_store {
                    self.layout.write
                } else {
                    self.layout.read
                };
                // Every `rs1 + k` for k in lo..=hi is inside the window
                // exactly when `rs1 + lo` is inside its first `len - (hi -
                // lo)` bytes: the span is below 2^13 and the window below
                // 2^64, so nothing wraps in between.
                let span = hi.wrapping_sub(lo) as u64;
                let local = self.checks;
                self.checks += 1;
                let k = if rs1 & 31 == 0 {
                    lit(lo.wrapping_sub(base as i64))
                } else {
                    format!(
                        "{}.wrapping_add({})",
                        x(rs1),
                        lit(lo.wrapping_sub(base as i64))
                    )
                };
                self.line(&format!(
                    "let k{local} = {k}; if k{local} >= {:#x} {{ break 'x A::X({EXIT_SLOW}, {pc:#x}); }}",
                    len.saturating_sub(span)
                ));
                let c = Check {
                    lo,
                    hi,
                    base,
                    local,
                };
                if let Some(slot) = self
                    .checked
                    .get_mut(usize::from(as_store))
                    .and_then(|k| k.get_mut(reg))
                {
                    *slot = Some(c);
                }
                c
            }
        };
        let d = c.base.wrapping_add(imm.wrapping_sub(c.lo) as u64);
        ("k", format!("m, k{}, {d:#x}", c.local))
    }

    /// Checks on entry to the region for the base registers it never
    /// writes: such a register holds its entry value everywhere in the
    /// region, so one check of the span of offsets of all the region's
    /// accesses through it covers every one of them. The check's result is
    /// a local `o<n>`; the first access of each straight-line run tests it
    /// and leaves there when it failed, as a check of the run's own span
    /// would (that one may pass where the region's fails: then native code
    /// leaves early, and the interpreter runs the accesses, which is always
    /// correct). Without them each run checks again, and LLVM hoists those
    /// loop-invariant compares into the function's entry one by one.
    ///
    /// Loads within the span of the stores share their check when every
    /// writable address is readable. Registers with fewer than two
    /// accesses, or a span wider than `REGION_SPAN_MAX`, keep the run
    /// checks.
    fn region_checks(&mut self, insns: &[usize], written: &[bool; 32]) -> Option<()> {
        const REGION_SPAN_MAX: i64 = 1 << 16;
        if self.layout.mask != 0 {
            return Some(());
        }
        // [load, store][register]: (lo, hi, count).
        let mut spans: [[Option<(i64, i64, u32)>; 32]; 2] = [[None; 32]; 2];
        for &i in insns {
            if let Some((rs1, off, write)) = memory_access(self.a.insns.get(i)?.op) {
                let reg = (rs1 & 31) as usize;
                if reg == 0 || written.get(reg).copied().unwrap_or(true) {
                    continue;
                }
                let slot = spans.get_mut(usize::from(write))?.get_mut(reg)?;
                *slot = Some(slot.map_or((off, off, 1), |(lo, hi, n)| {
                    (lo.min(off), hi.max(off), n.saturating_add(1))
                }));
            }
        }
        let stores_cover_loads = window_within(self.layout.write, self.layout.read);
        for reg in 1..32usize {
            let span = |kind: usize| spans.get(kind).and_then(|k| k.get(reg)).copied().flatten();
            let (loads, stores) = (span(0), span(1));
            let count = loads
                .map_or(0, |l| l.2)
                .saturating_add(stores.map_or(0, |s| s.2));
            if count < 2 {
                continue;
            }
            let narrow = |s: Option<(i64, i64, u32)>| {
                s.filter(|&(lo, hi, _)| hi.wrapping_sub(lo) <= REGION_SPAN_MAX)
            };
            let store_check =
                narrow(stores).map(|(lo, hi, _)| self.region_check(reg, lo, hi, true));
            let loads_with_stores = stores_cover_loads
                && matches!((loads, stores), (Some(l), Some(s)) if s.0 <= l.0 && l.1 <= s.1);
            let load_check = match store_check {
                Some(c) if loads_with_stores => Some(c),
                _ => narrow(loads).map(|(lo, hi, _)| self.region_check(reg, lo, hi, false)),
            };
            *self.region_checked.get_mut(0)?.get_mut(reg)? = load_check;
            *self.region_checked.get_mut(1)?.get_mut(reg)? = store_check;
        }
        Some(())
    }

    /// Write a region check of offsets `lo..=hi` through `reg` against the
    /// store (`write`) or load window (see [`Self::access`] for the
    /// arithmetic).
    fn region_check(&mut self, reg: usize, lo: i64, hi: i64, write: bool) -> Check {
        let (base, len) = if write {
            self.layout.write
        } else {
            self.layout.read
        };
        let span = hi.wrapping_sub(lo) as u64;
        let local = self.checks;
        self.checks += 1;
        self.region_lines.push(format!(
            "let k{local} = x{reg}.wrapping_add({}); let o{local} = k{local} < {:#x};",
            lit(lo.wrapping_sub(base as i64)),
            len.saturating_sub(span)
        ));
        Check {
            lo,
            hi,
            base,
            local,
        }
    }

    /// The offset spans `(lo, hi)` of the loads and of the stores through
    /// `rs1` from the instruction at `self.pos` to the end of its
    /// straight-line run (the next label, or an instruction that doesn't
    /// fall through), up to and including the first instruction that writes
    /// `rs1`: what a check made there can cover. (The JIT's `check_spans`.)
    fn check_spans(&self, rs1: u32) -> (Option<Span>, Option<Span>) {
        let mut spans: [Option<Span>; 2] = [None, None];
        let Some(run) = self
            .a
            .regions
            .get(self.region)
            .and_then(|r| r.insns.get(self.pos..))
        else {
            return (None, None);
        };
        let mut prev: Option<&Insn> = None;
        for &i in run {
            let Some(ins) = self.a.insns.get(i) else {
                break;
            };
            if let Some(p) = prev
                && (p.op.is_terminator()
                    || p.funcblock
                    || p.pc.wrapping_add(p.len) != ins.pc
                    || self.labels.contains(&ins.pc))
            {
                break;
            }
            if let Some((base, off, write)) = memory_access(ins.op)
                && base == rs1
                && let Some(span) = spans.get_mut(usize::from(write))
            {
                *span = Some(span.map_or((off, off), |(lo, hi)| (lo.min(off), hi.max(off))));
            }
            if operands(ins.op).xw == Some(rs1) {
                break;
            }
            prev = Some(ins);
        }
        let [loads, stores] = spans;
        (loads, stores)
    }

    /// Emit one instruction. Returns whether control falls through.
    fn instruction(&mut self, ins: &Insn) -> Option<bool> {
        let next = ins.pc.wrapping_add(ins.len);
        let pc = ins.pc;
        let slow = format!("break 'x A::X({EXIT_SLOW}, {pc:#x})");
        match ins.op {
            JOp::Nop => {}
            JOp::Dyncall => {
                // Leave with every register stored and the block counted;
                // the dispatcher runs the handler and re-enters at `next`.
                self.leave(EXIT_DYNCALL, pc);
                return Some(false);
            }
            JOp::Fence => {
                self.line("std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);");
            }
            JOp::Const { rd, value } => {
                if rd & 31 != 0 {
                    self.line(&format!("{} = {};", x(rd), lit(value as i64)));
                }
            }
            JOp::AluImm { op, rd, rs1, imm } => {
                if rd & 31 != 0 {
                    let v = alu_imm(op, &x(rs1), imm);
                    self.line(&format!("{} = {v};", x(rd)));
                }
            }
            JOp::Alu { op, rd, rs1, rs2 } => {
                if rd & 31 != 0 {
                    let v = alu(op, &x(rs1), &x(rs2));
                    self.line(&format!("{} = {v};", x(rd)));
                }
            }
            JOp::Unary { op, rd, rs1 } => {
                if rd & 31 != 0 {
                    let v = unary(op, &x(rs1));
                    self.line(&format!("{} = {v};", x(rd)));
                }
            }
            JOp::Load {
                width,
                signed,
                rd,
                rs1,
                imm,
            } => {
                let (k, a) = self.access(rs1, imm, false, pc);
                match rd.filter(|&rd| rd & 31 != 0) {
                    Some(rd) => {
                        let v = match (width, signed) {
                            (1, true) => "v[0] as i8 as i64 as u64",
                            (1, false) => "u64::from(v[0])",
                            (2, true) => "i16::from_le_bytes(v) as i64 as u64",
                            (2, false) => "u64::from(u16::from_le_bytes(v))",
                            (4, true) => "i32::from_le_bytes(v) as i64 as u64",
                            (4, false) => "u64::from(u32::from_le_bytes(v))",
                            _ => "u64::from_le_bytes(v)",
                        };
                        self.line(&format!(
                            "let Some(v) = ld{k}::<{width}>({a}) else {{ {slow} }}; {} = {v};",
                            x(rd)
                        ));
                    }
                    // The access is made (and may fault), the value dropped.
                    None => self.line(&format!("if ld{k}::<{width}>({a}).is_none() {{ {slow} }}")),
                }
            }
            JOp::Store {
                width,
                rs1,
                rs2,
                imm,
            } => {
                let (k, a) = self.access(rs1, imm, true, pc);
                let v = match width {
                    1 => format!("[{} as u8]", x(rs2)),
                    2 => format!("({} as u16).to_le_bytes()", x(rs2)),
                    4 => format!("({} as u32).to_le_bytes()", x(rs2)),
                    _ => format!("{}.to_le_bytes()", x(rs2)),
                };
                self.line(&format!("if !st{k}::<{width}>({a}, {v}) {{ {slow} }}"));
            }
            JOp::Branch {
                cond,
                rs1,
                rs2,
                target,
                check,
            } => {
                let a = x(rs1);
                let b = rs2.map_or_else(|| "0u64".to_owned(), x);
                let test = match cond {
                    Cond::Eq => format!("{a} == {b}"),
                    Cond::Ne => format!("{a} != {b}"),
                    Cond::Lt => format!("({a} as i64) < ({b} as i64)"),
                    Cond::Ge => format!("({a} as i64) >= ({b} as i64)"),
                    Cond::Ltu => format!("{a} < {b}"),
                    Cond::Geu => format!("{a} >= {b}"),
                };
                self.line(&format!("if {test} {{"));
                if check {
                    self.check_counter(target);
                }
                self.enter_block(target);
                self.line("}");
                self.enter_block(next);
                return Some(false);
            }
            JOp::Jal { rd, target } => {
                if rd & 31 != 0 {
                    self.line(&format!("{} = {next:#x};", x(rd)));
                }
                self.check_counter(target);
                self.enter_block(target);
                return Some(false);
            }
            JOp::Jalr { rd, rs1, imm } => {
                // The target is computed before the link register is written.
                self.line(&format!("let t = {} & !1;", Self::addr(rs1, imm)));
                if rd & 31 != 0 {
                    self.line(&format!("{} = {next:#x};", x(rd)));
                }
                if !COMPRESSED {
                    self.line(&format!(
                        "if t & 3 != 0 {{ break 'x A::X({EXIT_MISALIGNED}, t); }}"
                    ));
                }
                // overflow_checked_jump, then the entry dispatch (or the
                // entry table, outside this region).
                if !self.layout.unlimited {
                    self.line(&format!("if c >= max {{ break 'x A::X({EXIT_HALT}, t); }}"));
                }
                self.line("pc = t; continue 'e;");
                return Some(false);
            }
            JOp::FLoad {
                width,
                rd,
                rs1,
                imm,
            } => {
                let (k, a) = self.access(rs1, imm, false, pc);
                let d = f(rd);
                let v = if width == 4 {
                    format!(
                        "{d} = ({d} & 0xFFFF_FFFF_0000_0000) | u64::from(u32::from_le_bytes(v));"
                    )
                } else {
                    format!("{d} = u64::from_le_bytes(v);")
                };
                self.line(&format!(
                    "let Some(v) = ld{k}::<{width}>({a}) else {{ {slow} }}; {v}"
                ));
            }
            JOp::FStore {
                width,
                rs1,
                rs2,
                imm,
            } => {
                let (k, a) = self.access(rs1, imm, true, pc);
                let v = if width == 4 {
                    format!("({} as u32).to_le_bytes()", f(rs2))
                } else {
                    format!("{}.to_le_bytes()", f(rs2))
                };
                self.line(&format!("if !st{k}::<{width}>({a}, {v}) {{ {slow} }}"));
            }
            JOp::FBin {
                op,
                dbl,
                rd,
                rs1,
                rs2,
            } => {
                let (d, a, b) = (f(rd), f(rs1), f(rs2));
                let code = match op {
                    FBin::SgnJ | FBin::SgnJN | FBin::SgnJX => {
                        // Bit operations on the raw patterns; a single's are
                        // in the low half.
                        let sign = if dbl {
                            "0x8000_0000_0000_0000u64"
                        } else {
                            "0x8000_0000u64"
                        };
                        let s = match op {
                            FBin::SgnJ => format!("({b} & {sign})"),
                            FBin::SgnJN => format!("(!{b} & {sign})"),
                            _ => format!("(({a} ^ {b}) & {sign})"),
                        };
                        let v = format!("(({a} & !{sign}) | {s})");
                        if dbl {
                            format!("{d} = {v};")
                        } else {
                            format!("{d} = ({d} & 0xFFFF_FFFF_0000_0000) | ({v} & 0xFFFF_FFFF);")
                        }
                    }
                    _ => {
                        let (va, vb) = if dbl {
                            (format!("fd({a})"), format!("fd({b})"))
                        } else {
                            (format!("fs({a})"), format!("fs({b})"))
                        };
                        let sfx = if dbl { "d" } else { "s" };
                        let v = match op {
                            FBin::Add => format!("{va} + {vb}"),
                            FBin::Sub => format!("{va} - {vb}"),
                            FBin::Mul => format!("{va} * {vb}"),
                            FBin::Div => format!("{va} / {vb}"),
                            FBin::Min => format!("fmin_{sfx}({va}, {vb})"),
                            _ => format!("fmax_{sfx}({va}, {vb})"),
                        };
                        set_fval(&d, dbl, &v)
                    }
                };
                self.line(&code);
            }
            JOp::FSqrt { dbl, rd, rs1 } => {
                let v = format!("{}.sqrt()", fval(rs1, dbl));
                self.line(&set_fval(&f(rd), dbl, &v));
            }
            JOp::FFma {
                op,
                dbl,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                let (a, b, c) = (fval(rs1, dbl), fval(rs2, dbl), fval(rs3, dbl));
                let v = match op {
                    FFma::Madd => format!("{a}.mul_add({b}, {c})"),
                    FFma::Msub => format!("{a}.mul_add({b}, -{c})"),
                    FFma::Nmsub => format!("(-{a}).mul_add({b}, {c})"),
                    FFma::Nmadd => format!("-({a}.mul_add({b}, {c}))"),
                };
                self.line(&set_fval(&f(rd), dbl, &v));
            }
            JOp::FCvtFF { to_dbl, rd, rs1 } => {
                // A NaN converts to the canonical NaN (the interpreter's
                // `FCvtFF`), not the host's sign- and payload-keeping one.
                let v = if to_dbl {
                    format!("cvt_d_s({})", fval(rs1, false))
                } else {
                    format!("cvt_s_d({})", fval(rs1, true))
                };
                self.line(&set_fval(&f(rd), to_dbl, &v));
            }
            JOp::FCmp {
                op,
                dbl,
                rd,
                rs1,
                rs2,
            } => {
                let (a, b) = (fval(rs1, dbl), fval(rs2, dbl));
                let cmp = match op {
                    FCmp::Eq => "==",
                    FCmp::Lt => "<",
                    FCmp::Le => "<=",
                };
                if rd & 31 != 0 {
                    self.line(&format!("{} = u64::from({a} {cmp} {b});", x(rd)));
                }
            }
            JOp::FToInt {
                kind,
                dbl,
                rm,
                rd,
                rs1,
            } => {
                let k = match kind {
                    FInt::W => "w",
                    FInt::Wu => "wu",
                    FInt::L => "l",
                    FInt::Lu => "lu",
                };
                let sfx = if dbl { "d" } else { "s" };
                // DYN: FCSR.frm, which native code can't change (CSR
                // instructions are interpreted).
                let mode = if rm == 7 {
                    "frm".to_owned()
                } else {
                    rm.to_string()
                };
                if rd & 31 != 0 {
                    self.line(&format!(
                        "{} = cvt_{k}_{sfx}({}, {mode});",
                        x(rd),
                        fval(rs1, dbl)
                    ));
                }
            }
            JOp::IntToF { kind, dbl, rd, rs1 } => {
                let a = x(rs1);
                let ty = if dbl { "f64" } else { "f32" };
                let v = match kind {
                    FInt::W => format!("(({a} as i32) as {ty})"),
                    FInt::Wu => format!("(({a} as u32) as {ty})"),
                    FInt::L => format!("(({a} as i64) as {ty})"),
                    FInt::Lu => format!("({a} as {ty})"),
                };
                self.line(&set_fval(&f(rd), dbl, &v));
            }
            JOp::FMvXF { dbl, rd, rs1 } => {
                if rd & 31 != 0 {
                    let v = if dbl {
                        f(rs1)
                    } else {
                        format!("sx({} as u32)", f(rs1))
                    };
                    self.line(&format!("{} = {v};", x(rd)));
                }
            }
            JOp::FClass { dbl, rd, rs1 } => {
                if rd & 31 != 0 {
                    let v = if dbl {
                        format!(
                            "fclass({}, 63, 52, 0x7FF, 0xF_FFFF_FFFF_FFFF, 1 << 51)",
                            f(rs1)
                        )
                    } else {
                        format!(
                            "fclass(u64::from({} as u32), 31, 23, 0xFF, 0x7F_FFFF, 1 << 22)",
                            f(rs1)
                        )
                    };
                    self.line(&format!("{} = {v};", x(rd)));
                }
            }
            JOp::FMvFX { dbl, rd, rs1 } => {
                let d = f(rd);
                if dbl {
                    self.line(&format!("{d} = {};", x(rs1)));
                } else {
                    self.line(&format!(
                        "{d} = ({d} & 0xFFFF_FFFF_0000_0000) | u64::from({} as u32);",
                        x(rs1)
                    ));
                }
            }
        }
        Some(true)
    }
}

/// The base register, offset and direction (store) of a guest memory
/// access.
fn memory_access(op: JOp) -> Option<(u32, i64, bool)> {
    match op {
        JOp::Load { rs1, imm, .. } | JOp::FLoad { rs1, imm, .. } => Some((rs1, imm, false)),
        JOp::Store { rs1, imm, .. } | JOp::FStore { rs1, imm, .. } => Some((rs1, imm, true)),
        _ => None,
    }
}

/// True if the window `inner` lies inside `outer` and isn't empty.
fn window_within(inner: (u64, u64), outer: (u64, u64)) -> bool {
    inner.1 != 0
        && inner.0 >= outer.0
        && inner
            .0
            .checked_add(inner.1)
            .zip(outer.0.checked_add(outer.1))
            .is_some_and(|(i, o)| i <= o)
}

/// FP register `r` as an `f32` (its low half) or `f64`.
fn fval(r: u32, dbl: bool) -> String {
    if dbl {
        format!("fd({})", f(r))
    } else {
        format!("fs({})", f(r))
    }
}

/// Write the `f32`/`f64` expression `v` to FP register `d` (a single into
/// the low half, keeping the upper one: no NaN-boxing, as the interpreter).
fn set_fval(d: &str, dbl: bool, v: &str) -> String {
    if dbl {
        format!("{d} = ({v}).to_bits();")
    } else {
        format!("{d} = ss({d}, {v});")
    }
}

/// The 32-bit form of `v` (a `u32` expression), sign-extended (RV64 `*W`).
fn word(v: &str) -> String {
    format!("sx({v})")
}

fn alu_imm(op: AluI, a: &str, imm: i64) -> String {
    // Shift amounts are in range already; the JIT's Cranelift masks them
    // to the operand width anyway, and so does `wrapping_shl`.
    let sh = imm as u32;
    match op {
        AluI::Add => format!("{a}.wrapping_add({})", lit(imm)),
        AluI::Slt => format!("u64::from(({a} as i64) < {imm}i64)"),
        AluI::Sltu => format!("u64::from({a} < {})", lit(imm)),
        AluI::Xor => format!("({a} ^ {})", lit(imm)),
        AluI::Or => format!("({a} | {})", lit(imm)),
        AluI::And => format!("({a} & {})", lit(imm)),
        AluI::Sll => format!("{a}.wrapping_shl({sh})"),
        AluI::Srl => format!("{a}.wrapping_shr({sh})"),
        AluI::Sra => format!("(({a} as i64).wrapping_shr({sh}) as u64)"),
        AluI::AddW => word(&format!("({a} as u32).wrapping_add({:#x}u32)", imm as u32)),
        AluI::SllW => word(&format!("({a} as u32).wrapping_shl({sh})")),
        AluI::SrlW => word(&format!("({a} as u32).wrapping_shr({sh})")),
        AluI::SraW => word(&format!("(({a} as i32).wrapping_shr({sh}) as u32)")),
        AluI::Ror => format!("{a}.rotate_right({sh})"),
        AluI::RorW => word(&format!("({a} as u32).rotate_right({sh})")),
        AluI::Bset => format!("({a} | {})", lit(1i64.wrapping_shl(sh))),
        AluI::Bclr => format!("({a} & {})", lit(!1i64.wrapping_shl(sh))),
        AluI::Binv => format!("({a} ^ {})", lit(1i64.wrapping_shl(sh))),
        AluI::Bext => format!("({a}.wrapping_shr({sh}) & 1)"),
        AluI::SlliUw => format!("u64::from({a} as u32).wrapping_shl({sh})"),
    }
}

fn unary(op: Unary, a: &str) -> String {
    match op {
        Unary::ZextB => format!("u64::from({a} as u8)"),
        Unary::SextB => format!("({a} as i8 as i64 as u64)"),
        Unary::ZextH => format!("u64::from({a} as u16)"),
        Unary::SextH => format!("({a} as i16 as i64 as u64)"),
        Unary::ZextW => format!("u64::from({a} as u32)"),
        Unary::Not => format!("!{a}"),
        Unary::Clz => format!("u64::from({a}.leading_zeros())"),
        Unary::Ctz => format!("u64::from({a}.trailing_zeros())"),
        Unary::Cpop => format!("u64::from({a}.count_ones())"),
        Unary::ClzW => format!("u64::from(({a} as u32).leading_zeros())"),
        Unary::CtzW => format!("u64::from(({a} as u32).trailing_zeros())"),
        Unary::CpopW => format!("u64::from(({a} as u32).count_ones())"),
        Unary::Rev8 => format!("{a}.swap_bytes()"),
        Unary::Brev8 => format!("{a}.swap_bytes().reverse_bits()"),
        Unary::OrcB => format!("orcb({a})"),
    }
}

fn alu(op: AluR, a: &str, b: &str) -> String {
    match op {
        AluR::Add => format!("{a}.wrapping_add({b})"),
        AluR::Sub => format!("{a}.wrapping_sub({b})"),
        // Shift amounts are taken modulo the width, like RISC-V.
        AluR::Sll => format!("{a}.wrapping_shl({b} as u32)"),
        AluR::Srl => format!("{a}.wrapping_shr({b} as u32)"),
        AluR::Sra => format!("(({a} as i64).wrapping_shr({b} as u32) as u64)"),
        AluR::Slt => format!("u64::from(({a} as i64) < ({b} as i64))"),
        AluR::Sltu => format!("u64::from({a} < {b})"),
        AluR::Xor => format!("({a} ^ {b})"),
        AluR::Or => format!("({a} | {b})"),
        AluR::And => format!("({a} & {b})"),
        AluR::Mul => format!("{a}.wrapping_mul({b})"),
        AluR::Mulh => format!("mulh({a}, {b})"),
        AluR::Mulhsu => format!("mulhsu({a}, {b})"),
        AluR::Mulhu => format!("mulhu({a}, {b})"),
        AluR::Div => format!("div({a}, {b})"),
        AluR::Divu => format!("divu({a}, {b})"),
        AluR::Rem => format!("rem({a}, {b})"),
        AluR::Remu => format!("remu({a}, {b})"),
        AluR::AddW => word(&format!("({a} as u32).wrapping_add({b} as u32)")),
        AluR::SubW => word(&format!("({a} as u32).wrapping_sub({b} as u32)")),
        AluR::SllW => word(&format!("({a} as u32).wrapping_shl({b} as u32)")),
        AluR::SrlW => word(&format!("({a} as u32).wrapping_shr({b} as u32)")),
        AluR::SraW => word(&format!("(({a} as i32).wrapping_shr({b} as u32) as u32)")),
        AluR::MulW => word(&format!("({a} as u32).wrapping_mul({b} as u32)")),
        AluR::DivW => format!("divw({a}, {b})"),
        AluR::DivuW => format!("divuw({a}, {b})"),
        AluR::RemW => format!("remw({a}, {b})"),
        AluR::RemuW => format!("remuw({a}, {b})"),
        AluR::ShAdd(n) => format!("{b}.wrapping_add({a} << {n})"),
        AluR::ShAddUw(n) => format!("{b}.wrapping_add(u64::from({a} as u32) << {n})"),
        AluR::Pack => format!("(u64::from({a} as u32) | ({b} << 32))"),
        AluR::PackH => format!("(({a} & 0xFF) | (({b} & 0xFF) << 8))"),
        AluR::PackW => word(&format!("((({a} as u32) & 0xFFFF) | (({b} as u32) << 16))")),
        AluR::Min => format!("(({a} as i64).min({b} as i64) as u64)"),
        AluR::Max => format!("(({a} as i64).max({b} as i64) as u64)"),
        AluR::Minu => format!("{a}.min({b})"),
        AluR::Maxu => format!("{a}.max({b})"),
        AluR::CzeroEqz => format!("(if {b} != 0 {{ {a} }} else {{ 0 }})"),
        AluR::CzeroNez => format!("(if {b} != 0 {{ 0 }} else {{ {a} }})"),
        AluR::Bset => format!("({a} | 1u64.wrapping_shl({b} as u32))"),
        AluR::Bclr => format!("({a} & !1u64.wrapping_shl({b} as u32))"),
        AluR::Binv => format!("({a} ^ 1u64.wrapping_shl({b} as u32))"),
        AluR::Bext => format!("({a}.wrapping_shr({b} as u32) & 1)"),
        AluR::Xnor => format!("!({a} ^ {b})"),
        AluR::Orn => format!("({a} | !{b})"),
        AluR::Andn => format!("({a} & !{b})"),
        // Rotate amounts are taken modulo the width, like RISC-V.
        AluR::Rol => format!("{a}.rotate_left({b} as u32)"),
        AluR::Ror => format!("{a}.rotate_right({b} as u32)"),
        AluR::RolW => word(&format!("({a} as u32).rotate_left({b} as u32)")),
        AluR::RorW => word(&format!("({a} as u32).rotate_right({b} as u32)")),
    }
}
