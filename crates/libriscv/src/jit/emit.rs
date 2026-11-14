//! Port of `asmjit/aj_emit.cpp`, generating Cranelift IR instead of asmjit
//! code for the regions of one execute segment.
//!
//! Every region becomes one function with Cranelift's `tail` calling
//! convention and the signature `(state, pc, counter, depth) -> (target,
//! counter)`. It switches on `pc` to the entry point, keeps the guest
//! registers it uses in SSA variables, and leaves in one of four ways:
//!
//! * a native tail call (`return_call`) to the region owning the next
//!   instruction, so loops and jumps never come back through the
//!   interpreter; indirect jumps go through the entry table
//!   (`return_call_indirect`);
//! * a guest return (`jalr x0, ra` or `t0`), which returns its target to
//!   the native caller: a guest call whose return address is in the
//!   caller's region is a native call (C++ asmjit's `emit_direct_call`),
//!   which continues there if the callee returned there (see
//!   `codegen::calls`). The trampoline continues other returns through the
//!   entry table;
//! * a return of [`DONE`] with an exit kind in the state block (the
//!   instruction limit, an instruction the JIT doesn't compile, a fault),
//!   which every native caller passes on;
//! * never by unwinding: faults are exits.
//!
//! # Instruction counting
//!
//! The JIT retires instructions exactly like the dispatcher. Entering a
//! block at `X` (taken branch, jump, fall-through after a branch, function
//! entry) adds the decoder's instruction count for `X`; checks happen at the
//! same block ends. When control leaves *mid-block* (into another region or
//! back to the interpreter), the count for the target is subtracted first,
//! since whoever continues adds it again on entry.
//!
//! # Memory safety
//!
//! Generated code only touches the state block, the 32 integer and 32
//! floating-point registers and the guest arena, and every arena access is
//! covered by the same range check as `Memory::read_*`/`write_*`, or with
//! an N-bit address space, masked with the arena size minus one (C++
//! `arena_index` in `aj_emit.cpp`); `CodeMemory` checks the arena is that
//! large. The windows are constants in the code (C++ `aj_machine_info`'s
//! baked arena bounds), and `libriscv-jit-rt` only enters the code when
//! they lie inside the machine's windows, which it validates against the
//! real arena length. See its crate documentation for the full contract.
//!
//! One check can cover several accesses: the first access through a base
//! register checks the whole span of offsets that the accesses through the
//! same register (and of the same kind, load or store) use until the end of
//! its straight-line run, or until the register changes. Each access's own
//! check is on its start address, so the span check passes exactly when
//! all of theirs would; when it fails, native code leaves at the first
//! access and the interpreter runs the rest of the block, as it would for
//! that access alone. (C++'s binary translator elides checks within the
//! arena's over-allocation of the first one instead, which lets accesses
//! near the window edges through; this keeps the interpreter's results.) Besides other region functions, it
//! only calls the pure floating-point helpers below (Cranelift's libcalls
//! for rounding and FMA on hosts without SSE4.1 or FMA).
//!
//! # Floating point
//!
//! F/D arithmetic uses Cranelift's operations as they are: results for
//! non-NaN inputs are the interpreter's (both use the host's IEEE
//! operations in round-to-nearest), but which NaN comes out follows the
//! host (NaN payloads are FCSR emulation's business). FMIN/FMAX with one
//! NaN operand give the other one and with two the canonical NaN, as
//! RISC-V defines, and FCVT to integers honours the
//! rounding mode and saturates (NaN to the maximum), exactly like the
//! interpreter.

use std::collections::HashMap;

use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir::condcodes::{CondCode, FloatCC, IntCC};
use cranelift_codegen::ir::immediates::{Ieee32, Ieee64, Offset32};
use cranelift_codegen::ir::types::{F32, F64, I8, I16, I32, I64};
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, Endianness, ExtFuncData, ExternalName, FuncRef, Function,
    InstBuilder, LibCall, MemFlagsData, SigRef, Signature, Type, UserExternalName, UserFuncName,
    Value,
};
use cranelift_codegen::isa::{CallConv, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{Context, RelocTarget};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use libriscv_jit_rt::{CodeMemory, JitState, MemoryLayout};

use crate::codegen::calls::{
    DEPTH, DONE, RegSet, Summary, direct_jalr_target, is_return, region_regs, summaries,
};
use crate::codegen::classify::{AluI, AluR, Cond, FBin, FCmp, FFma, FInt, JOp, Unary, operands};
use crate::codegen::region::{Analysis, Insn, RegionId, block_icount};
use crate::decoder_cache::{COMPRESSED, DecodedSegment, SHIFT};
use crate::native_code::{
    EXIT_DYNCALL, EXIT_HALT, EXIT_INDIRECT, EXIT_JUMP, EXIT_MISALIGNED, EXIT_SLOW,
};

/// Functions are placed at this alignment in the code image.
const FUNCTION_ALIGN: usize = 64;

/// The compiled segment: code memory plus the decoder indices to patch.
#[derive(Debug)]
pub(crate) struct Compiled {
    pub code: CodeMemory,
    pub patch: Vec<usize>,
}

/// A compiled and linked segment that isn't in executable memory yet: the
/// code image with its calls between functions resolved, but not its
/// libcalls. [`Image::load`] maps it for this process; the JIT cache saves
/// it as an object file (`jit_cache`), so that the same bytes run later.
#[derive(Debug, Clone)]
pub(crate) struct Image {
    pub bytes: Vec<u8>,
    /// Offset of the trampoline.
    pub trampoline: usize,
    /// Per decoder entry, the offset of the function entered there, or
    /// [`CodeMemory::NO_ENTRY`].
    pub entries: Vec<u32>,
    /// The decoder entries to patch: block starts with an entry.
    pub patch: Vec<usize>,
    /// The libcalls, each a 32-bit PC-relative reference to a GOT slot
    /// holding the helper's address (the code is position-independent).
    pub libcalls: Vec<LibcallSite>,
    pub layout: MemoryLayout,
}

/// A reference to the GOT slot of a libcall (`X86GOTPCRel4`): the 4 bytes
/// at `offset` are `slot + addend - offset`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LibcallSite {
    pub offset: usize,
    pub addend: i64,
    pub call: LibCall,
}

impl Image {
    /// Copy the image into executable memory, with a GOT for its libcalls
    /// appended, pointing at this process's helpers.
    pub(crate) fn load(self) -> Option<Compiled> {
        let mut image = self.bytes;
        if !self.libcalls.is_empty() {
            let mut slots: Vec<(LibCall, usize)> = Vec::new();
            image.resize(image.len().next_multiple_of(8), 0);
            for site in &self.libcalls {
                let slot = match slots.iter().find(|(c, _)| *c == site.call) {
                    Some(&(_, slot)) => slot,
                    None => {
                        let slot = image.len();
                        image.extend_from_slice(&libcall_address(site.call)?.to_le_bytes());
                        slots.push((site.call, slot));
                        slot
                    }
                };
                let value = i64::try_from(slot)
                    .ok()?
                    .checked_add(site.addend)?
                    .checked_sub(i64::try_from(site.offset).ok()?)?;
                let value = i32::try_from(value).ok()?;
                image
                    .get_mut(site.offset..site.offset + 4)?
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
        let code = CodeMemory::new(&image, self.trampoline, &self.entries, self.layout).ok()?;
        Some(Compiled {
            code,
            patch: self.patch,
        })
    }
}

/// The C library function implementing `call` (the names `cranelift-module`
/// uses), for saved code: see [`Image::libcalls`].
pub(crate) fn libcall_symbol(call: LibCall) -> Option<&'static str> {
    Some(match call {
        LibCall::CeilF32 => "ceilf",
        LibCall::CeilF64 => "ceil",
        LibCall::FloorF32 => "floorf",
        LibCall::FloorF64 => "floor",
        LibCall::TruncF32 => "truncf",
        LibCall::TruncF64 => "trunc",
        LibCall::NearestF32 => "nearbyintf",
        LibCall::NearestF64 => "nearbyint",
        LibCall::FmaF32 => "fmaf",
        LibCall::FmaF64 => "fma",
        _ => return None,
    })
}

/// Everything about this build that the code generated for a segment
/// depends on besides the segment and the memory layout: the host ISA and
/// its features, the libriscv version, its extension features, the layout
/// of the state block, and the source of the code generator itself.
/// Cached code (`jit_cache`) is only used by builds with the same
/// fingerprint.
pub(crate) fn build_fingerprint() -> Option<u64> {
    static FINGERPRINT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let isa = host_isa()?;
        let mut text = format!(
            "libriscv {} {} {} {:?} {:?}",
            env!("CARGO_PKG_VERSION"),
            isa.triple(),
            isa.flags(),
            isa.isa_flags()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            [
                cfg!(feature = "ext-a"),
                cfg!(feature = "ext-c"),
                cfg!(feature = "ext-f"),
                cfg!(feature = "ext-v"),
                cfg!(feature = "debug-checks"),
            ],
        );
        for offset in [
            JitState::OFFSET_REGS,
            JitState::OFFSET_FREGS,
            JitState::OFFSET_ARENA,
            JitState::OFFSET_TABLE,
            JitState::OFFSET_TABLE_LEN,
            JitState::OFFSET_MAX,
            JitState::OFFSET_FRM,
            JitState::OFFSET_COUNTER,
            JitState::OFFSET_EXIT_PC,
            JitState::OFFSET_EXIT_KIND,
        ] {
            text.push_str(&format!(" {offset}"));
        }
        // What decides the generated code, besides `codegen::SHARED_SOURCES`.
        Some(crate::codegen::fingerprint(
            text,
            &[include_str!("emit.rs"), include_str!("mod.rs")],
        ))
    })
}

/// The host ISA, or `None` where the JIT isn't supported. Only x86-64 is
/// enabled for now: AArch64 needs an explicit instruction-cache flush after
/// writing code, which `libriscv-jit-rt` doesn't do yet.
fn host_isa() -> Option<OwnedTargetIsa> {
    if !cfg!(target_arch = "x86_64") {
        return None;
    }
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").ok()?;
    // Position-independent code (libcalls through a GOT), so that the same
    // code can also be saved and linked elsewhere (`jit_cache`).
    flags.set("is_pic", "true").ok()?;
    // Cranelift's x64 tail calls rely on frame pointers.
    flags.set("preserve_frame_pointers", "true").ok()?;
    flags
        .set(
            "enable_verifier",
            if cfg!(debug_assertions) {
                "true"
            } else {
                "false"
            },
        )
        .ok()?;
    #[cfg(test)]
    if BASELINE_ISA.load(std::sync::atomic::Ordering::Relaxed) {
        // Plain x86-64 (SSE2): rounding and FMA become libcalls.
        let isa = cranelift_codegen::isa::lookup_by_name("x86_64-unknown-linux-gnu").ok()?;
        return isa.finish(settings::Flags::new(flags)).ok();
    }
    let isa = cranelift_native::builder().ok()?;
    isa.finish(settings::Flags::new(flags)).ok()
}

/// Tests: compile for baseline x86-64 instead of the host's features.
#[cfg(test)]
pub(crate) static BASELINE_ISA: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// A region function: `(state, pc, counter, depth) -> (target, counter)`,
/// where `target` is where a guest return went (to the native caller,
/// which continues if it is its return address), or [`DONE`] if the code
/// left for the host (the exit is in the state block), and `depth` the
/// native calls in progress (see [`DEPTH`]).
fn tail_signature() -> Signature {
    let mut sig = Signature::new(CallConv::Tail);
    for _ in 0..4 {
        sig.params.push(AbiParam::new(I64));
    }
    for _ in 0..2 {
        sig.returns.push(AbiParam::new(I64));
    }
    sig
}

/// What a relocation refers to.
#[derive(Debug, Clone, Copy)]
enum Target {
    /// Another function of the image.
    Function(usize),
    /// A host helper (through the GOT).
    LibCall(LibCall),
}

/// One function's machine code and the calls in it that need linking.
struct Object {
    bytes: Vec<u8>,
    /// `(offset, kind, addend, target)`
    relocs: Vec<(usize, Reloc, i64, Target)>,
}

// Cranelift's libcalls, for hosts without the instructions (SSE4.1's
// `roundss`/`roundsd`, FMA). Plain Rust: the same results as the
// interpreter's.
extern "C" fn ceil_f32(x: f32) -> f32 {
    x.ceil()
}
extern "C" fn ceil_f64(x: f64) -> f64 {
    x.ceil()
}
extern "C" fn floor_f32(x: f32) -> f32 {
    x.floor()
}
extern "C" fn floor_f64(x: f64) -> f64 {
    x.floor()
}
extern "C" fn trunc_f32(x: f32) -> f32 {
    x.trunc()
}
extern "C" fn trunc_f64(x: f64) -> f64 {
    x.trunc()
}
extern "C" fn nearest_f32(x: f32) -> f32 {
    x.round_ties_even()
}
extern "C" fn nearest_f64(x: f64) -> f64 {
    x.round_ties_even()
}
extern "C" fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}
extern "C" fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// The address of the helper implementing `call`, if there is one.
fn libcall_address(call: LibCall) -> Option<u64> {
    type F1 = extern "C" fn(f32) -> f32;
    type D1 = extern "C" fn(f64) -> f64;
    type F3 = extern "C" fn(f32, f32, f32) -> f32;
    type D3 = extern "C" fn(f64, f64, f64) -> f64;
    let address = match call {
        LibCall::CeilF32 => ceil_f32 as F1 as usize,
        LibCall::CeilF64 => ceil_f64 as D1 as usize,
        LibCall::FloorF32 => floor_f32 as F1 as usize,
        LibCall::FloorF64 => floor_f64 as D1 as usize,
        LibCall::TruncF32 => trunc_f32 as F1 as usize,
        LibCall::TruncF64 => trunc_f64 as D1 as usize,
        LibCall::NearestF32 => nearest_f32 as F1 as usize,
        LibCall::NearestF64 => nearest_f64 as D1 as usize,
        LibCall::FmaF32 => fma_f32 as F3 as usize,
        LibCall::FmaF64 => fma_f64 as D3 as usize,
        _ => return None,
    };
    u64::try_from(address).ok()
}

fn compile_function(ctx: &mut Context, isa: &OwnedTargetIsa) -> Option<Object> {
    let mut ctrl = ControlPlane::default();
    let (bytes, raw) = {
        let code = ctx.compile(isa.as_ref(), &mut ctrl).ok()?;
        (code.code_buffer().to_vec(), code.buffer.relocs().to_vec())
    };
    let mut relocs = Vec::new();
    for r in &raw {
        let target = match &r.target {
            RelocTarget::ExternalName(ExternalName::User(name)) => {
                Target::Function(ctx.func.params.user_named_funcs().get(*name)?.index as usize)
            }
            RelocTarget::ExternalName(ExternalName::LibCall(call)) => {
                libcall_address(*call)?;
                Target::LibCall(*call)
            }
            _ => return None,
        };
        relocs.push((r.offset as usize, r.kind, r.addend, target));
    }
    Some(Object { bytes, relocs })
}

/// Apply a call relocation in `image`: the call at `at` refers to `what`
/// (both image offsets). Mirrors `cranelift-jit`'s `CompiledBlob::perform_relocations`.
fn relocate(image: &mut [u8], at: usize, kind: Reloc, what: i64) -> Option<()> {
    let pcrel = what.checked_sub(i64::try_from(at).ok()?)?;
    match kind {
        Reloc::X86CallPCRel4 | Reloc::X86CallPLTRel4 => {
            let v = i32::try_from(pcrel).ok()?;
            image.get_mut(at..at + 4)?.copy_from_slice(&v.to_le_bytes());
        }
        Reloc::Arm64Call => {
            let diff = pcrel >> 2;
            if !(-(1 << 25)..(1 << 25)).contains(&diff) {
                return None;
            }
            let slot = image.get_mut(at..at + 4)?;
            let ins = u32::from_le_bytes(slot.try_into().ok()?);
            let ins = (ins & 0xFC00_0000) | (diff as u32 & 0x03FF_FFFF);
            slot.copy_from_slice(&ins.to_le_bytes());
        }
        _ => return None,
    }
    Some(())
}

/// Compile every region of `a` and link them into an [`Image`].
/// `layout`: see [`compile`](super::compile).
pub(crate) fn compile(exec: &DecodedSegment, a: &Analysis, layout: MemoryLayout) -> Option<Image> {
    if a.regions.is_empty() {
        return None;
    }
    let isa = host_isa()?;
    let summaries = summaries(exec, a)?;
    let mut ctx = Context::new();
    let mut fctx = FunctionBuilderContext::new();
    let mut objects = Vec::with_capacity(a.regions.len() + 1);
    for r in 0..a.regions.len() {
        ctx.clear();
        ctx.func = Function::with_name_signature(UserFuncName::user(0, r as u32), tail_signature());
        let segment = Segment {
            exec,
            a,
            layout,
            summaries: &summaries,
        };
        RegionEmitter::emit(&segment, r, &mut ctx.func, &mut fctx, &isa)?;
        objects.push(compile_function(&mut ctx, &isa)?);
    }
    // The trampoline: called from Rust with the C calling convention as
    // `(state, function, pc, counter)`, it enters the `tail` world.
    ctx.clear();
    ctx.func = trampoline(exec, &isa, &mut fctx);
    objects.push(compile_function(&mut ctx, &isa)?);

    // Link: lay out the functions, then resolve the direct calls. Libcalls
    // stay unresolved: they go through a GOT, which whoever places the
    // image provides.
    let mut image = Vec::new();
    let mut offsets = Vec::with_capacity(objects.len());
    for obj in &objects {
        image.resize(image.len().next_multiple_of(FUNCTION_ALIGN), 0);
        offsets.push(image.len());
        image.extend_from_slice(&obj.bytes);
    }
    let mut libcalls = Vec::new();
    for (obj, &base) in objects.iter().zip(&offsets) {
        for &(offset, kind, addend, target) in &obj.relocs {
            match target {
                Target::Function(f) => {
                    let what = i64::try_from(*offsets.get(f)?).ok()?;
                    relocate(&mut image, base + offset, kind, what.checked_add(addend)?)?;
                }
                Target::LibCall(call) if kind == Reloc::X86GOTPCRel4 => {
                    libcalls.push(LibcallSite {
                        offset: base + offset,
                        addend,
                        call,
                    });
                }
                Target::LibCall(_) => return None,
            }
        }
    }

    let mut entries = vec![CodeMemory::NO_ENTRY; exec.cache.len()];
    for (r, region) in a.regions.iter().enumerate() {
        let offset = u32::try_from(*offsets.get(r)?).ok()?;
        for &pc in &region.entries {
            *entries.get_mut(exec.index(pc))? = offset;
        }
    }
    let trampoline = *offsets.last()?;
    let patch = a
        .block_starts
        .iter()
        .copied()
        .filter(|&i| entries.get(i).is_some_and(|&e| e != CodeMemory::NO_ENTRY))
        .collect();
    Some(Image {
        bytes: image,
        trampoline,
        entries,
        patch,
        libcalls,
        layout,
    })
}

/// The trampoline: calls the region function, and while it returns a
/// guest return target (a return that no native caller took), continues
/// there through the entry table, as an indirect jump would. A target
/// without an entry leaves with `EXIT_INDIRECT`.
fn trampoline(
    exec: &DecodedSegment,
    isa: &OwnedTargetIsa,
    fctx: &mut FunctionBuilderContext,
) -> Function {
    let mut sig = Signature::new(isa.default_call_conv());
    for _ in 0..4 {
        sig.params.push(AbiParam::new(I64));
    }
    let mut func = Function::with_name_signature(UserFuncName::user(1, 0), sig);
    let mut b = FunctionBuilder::new(&mut func, fctx);
    let tail = b.import_signature(tail_signature());
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    let run = b.create_block();
    for _ in 0..3 {
        b.append_block_param(run, I64);
    }
    let returned = b.create_block();
    let lookup = b.create_block();
    let miss = b.create_block();
    let done = b.create_block();
    for blk in [returned, lookup, miss] {
        b.append_block_param(blk, I64);
        b.append_block_param(blk, I64);
    }
    b.switch_to_block(entry);
    let p = b.block_params(entry).to_vec();
    let &[state, function, pc, counter] = &p[..] else {
        return func;
    };
    b.ins().jump(run, &args(&[function, pc, counter]));

    b.switch_to_block(run);
    let p = b.block_params(run).to_vec();
    if let &[function, pc, counter] = &p[..] {
        let zero = b.ins().iconst(I64, 0);
        let call = b
            .ins()
            .call_indirect(tail, function, &[state, pc, counter, zero]);
        let r = b.inst_results(call).to_vec();
        if let &[target, counter] = &r[..] {
            let is_done = b.ins().icmp_imm_s(IntCC::Equal, target, DONE as i64);
            b.ins()
                .brif(is_done, done, &[], returned, &args(&[target, counter]));
        }
    }

    b.switch_to_block(returned);
    let p = b.block_params(returned).to_vec();
    if let &[target, counter] = &p[..] {
        // slot = (target >> SHIFT) - (begin >> SHIFT), as `DecodedSegment::index`
        let shifted = b.ins().ushr_imm_s(target, i64::from(SHIFT));
        let begin = (exec.exec_begin() >> SHIFT) as i64;
        let slot = b.ins().iadd_imm_s(shifted, begin.wrapping_neg());
        let len = b
            .ins()
            .load(I64, host_flags(), state, JitState::OFFSET_TABLE_LEN);
        let in_table = b.ins().icmp(IntCC::UnsignedLessThan, slot, len);
        let table = b
            .ins()
            .load(I64, host_flags(), state, JitState::OFFSET_TABLE);
        let byte_off = b.ins().ishl_imm_s(slot, 3);
        let addr = b.ins().iadd(table, byte_off);
        b.ins().brif(
            in_table,
            lookup,
            &args(&[target, counter]),
            miss,
            &args(&[target, counter]),
        );
        b.switch_to_block(lookup);
        let p = b.block_params(lookup).to_vec();
        if let &[target, counter] = &p[..] {
            let function = b.ins().load(I64, host_flags(), addr, 0);
            let next = b.create_block();
            b.ins()
                .brif(function, next, &[], miss, &args(&[target, counter]));
            b.switch_to_block(next);
            b.ins().jump(run, &args(&[function, target, counter]));
        }
    }

    b.switch_to_block(miss);
    let p = b.block_params(miss).to_vec();
    if let &[target, counter] = &p[..] {
        let kind = b.ins().iconst(I64, EXIT_INDIRECT as i64);
        b.ins()
            .store(host_flags(), kind, state, JitState::OFFSET_EXIT_KIND);
        b.ins()
            .store(host_flags(), target, state, JitState::OFFSET_EXIT_PC);
        b.ins()
            .store(host_flags(), counter, state, JitState::OFFSET_COUNTER);
    }
    b.ins().return_(&[]);

    b.switch_to_block(done);
    b.ins().return_(&[]);
    b.seal_all_blocks();
    b.finalize(isa.frontend_config());
    func
}

fn guest_flags() -> MemFlagsData {
    MemFlagsData::new()
        .with_notrap()
        .with_endianness(Endianness::Little)
}

fn host_flags() -> MemFlagsData {
    MemFlagsData::trusted()
}

fn args(values: &[Value]) -> Vec<BlockArg> {
    values.iter().copied().map(BlockArg::Value).collect()
}

/// Emits one region.
struct RegionEmitter<'a, 'f> {
    exec: &'a DecodedSegment,
    a: &'a Analysis,
    region: RegionId,
    /// Guest address mask (or 0) and the windows checked against.
    layout: MemoryLayout,
    b: FunctionBuilder<'f>,
    regs: [Variable; 32],
    /// The FP registers, as raw 64-bit patterns.
    fregs: [Variable; 32],
    counter: Variable,
    /// Registers written anywhere in the region (stored on every exit).
    written: [bool; 32],
    fwritten: [bool; 32],
    state: Value,
    regs_ptr: Value,
    fregs_ptr: Value,
    arena: Value,
    labels: HashMap<u64, Block>,
    /// Position in the region's instruction list of the instruction being
    /// emitted.
    pos: usize,
    /// `[load, store][base register]`: the span of offsets an earlier check
    /// in this straight-line run covers for accesses through that register.
    checked: [[Option<Span>; 32]; 2],
    /// `(kind, pc, counter)`: store the registers, then leave.
    exit: Block,
    /// `(kind, pc, counter)`: leave; the registers are already stored.
    exit_stored: Block,
    /// Target region -> `(pc, counter)` block that tail-calls it.
    xfer: HashMap<RegionId, Block>,
    /// `(target, counter)`: an indirect jump through the entry table.
    indirect: Option<Block>,
    /// `(target, counter)`: store the registers, then return `target` to
    /// the native caller (a guest return, or on after a callee that left
    /// for the host: `DONE`).
    ret: Option<Block>,
    funcs: HashMap<RegionId, FuncRef>,
    tail_sig: SigRef,
    /// The native calls in progress (see [`DEPTH`]).
    depth: Value,
    /// Every region's [`Summary`].
    summaries: &'a [Summary],
    /// The registers this region uses and writes.
    used_set: RegSet,
    written_set: RegSet,
    /// Index in `a.insns` of the instruction being emitted.
    index: usize,
}

/// What every region of a segment is compiled from.
struct Segment<'a> {
    exec: &'a DecodedSegment,
    a: &'a Analysis,
    /// Guest address mask (or 0) and the windows checked against.
    layout: MemoryLayout,
    /// Every region's [`Summary`].
    summaries: &'a [Summary],
}

impl<'a, 'f> RegionEmitter<'a, 'f> {
    fn emit(
        segment: &Segment<'a>,
        region: RegionId,
        func: &'f mut Function,
        fctx: &'f mut FunctionBuilderContext,
        isa: &OwnedTargetIsa,
    ) -> Option<()> {
        let &Segment {
            exec,
            a,
            layout,
            summaries,
        } = segment;
        let r = a.regions.get(region)?;
        // x0 is a constant: never loaded, never stored.
        let (used_set, written_set) = region_regs(a, region)?;
        let bit = |set: RegSet, n: usize| set >> n & 1 != 0;
        let used: [bool; 32] = std::array::from_fn(|n| n != 0 && bit(used_set, n));
        let written: [bool; 32] = std::array::from_fn(|n| n != 0 && bit(written_set, n));
        let fused: [bool; 32] = std::array::from_fn(|n| bit(used_set, 32 + n));
        let fwritten: [bool; 32] = std::array::from_fn(|n| bit(written_set, 32 + n));

        let mut b = FunctionBuilder::new(func, fctx);
        let regs: [Variable; 32] = std::array::from_fn(|_| b.declare_var(I64));
        let fregs: [Variable; 32] = std::array::from_fn(|_| b.declare_var(I64));
        let counter = b.declare_var(I64);
        let tail_sig = b.import_signature(tail_signature());

        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let (state, pc_in, counter_in, depth) = match b.block_params(entry) {
            &[s, p, c, d] => (s, p, c, d),
            _ => return None,
        };
        let ld =
            |b: &mut FunctionBuilder<'_>, off: i32| b.ins().load(I64, host_flags(), state, off);
        let regs_ptr = ld(&mut b, JitState::OFFSET_REGS);
        let fregs_ptr = ld(&mut b, JitState::OFFSET_FREGS);
        let arena = ld(&mut b, JitState::OFFSET_ARENA);
        for (n, (&u, &var)) in used.iter().zip(&regs).enumerate() {
            if u {
                let v = b.ins().load(I64, host_flags(), regs_ptr, (n * 8) as i32);
                b.def_var(var, v);
            }
        }
        for (n, (&u, &var)) in fused.iter().zip(&fregs).enumerate() {
            if u {
                let v = b.ins().load(I64, host_flags(), fregs_ptr, (n * 8) as i32);
                b.def_var(var, v);
            }
        }
        b.def_var(counter, counter_in);

        let exit = b.create_block();
        let exit_stored = b.create_block();
        for blk in [exit, exit_stored] {
            for _ in 0..3 {
                b.append_block_param(blk, I64);
            }
            b.set_cold_block(blk);
        }

        let mut e = RegionEmitter {
            exec,
            a,
            region,
            layout,
            b,
            regs,
            fregs,
            counter,
            written,
            fwritten,
            state,
            regs_ptr,
            fregs_ptr,
            arena,
            labels: HashMap::new(),
            pos: 0,
            checked: [[None; 32]; 2],
            exit,
            exit_stored,
            xfer: HashMap::new(),
            indirect: None,
            ret: None,
            funcs: HashMap::new(),
            tail_sig,
            depth,
            summaries,
            used_set,
            written_set,
            index: 0,
        };

        // Entry dispatch: one block per entry point adds the block's count
        // (nothing without the limit, `MemoryLayout::unlimited`).
        let mut switch = Switch::new();
        let mut enters = Vec::with_capacity(r.entries.len());
        for &pc in &r.entries {
            let label = e.b.create_block();
            e.labels.insert(pc, label);
            let enter = e.b.create_block();
            switch.set_entry(u128::from(pc), enter);
            enters.push((pc, enter, label));
        }
        let unknown = e.b.create_block();
        switch.emit(&mut e.b, pc_in, unknown);
        for (pc, enter, label) in enters {
            e.b.switch_to_block(enter);
            let k = if layout.unlimited {
                0
            } else {
                block_icount(exec, pc)
            };
            let c = e.b.ins().iadd_imm_s(counter_in, k as i64);
            e.b.def_var(counter, c);
            e.b.ins().jump(label, &[]);
        }
        e.b.switch_to_block(unknown);
        let kind = e.b.ins().iconst(I64, EXIT_JUMP as i64);
        e.b.ins().jump(exit, &args(&[kind, pc_in, counter_in]));

        e.body(&r.insns)?;
        e.emit_exits();
        e.b.seal_all_blocks();
        e.b.finalize(isa.frontend_config());
        Some(())
    }

    fn insn(&self, i: usize) -> Option<&'a Insn> {
        self.a.insns.get(i)
    }

    /// Integer register `r`. `x0` is the constant zero (C++ asmjit's
    /// `get(0)`): nothing ever writes it, in the register file either.
    fn reg(&mut self, r: u32) -> Value {
        let r = (r & 31) as usize;
        if r == 0 {
            return self.b.ins().iconst(I64, 0);
        }
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        let var = self.regs[r];
        self.b.use_var(var)
    }

    /// Write integer register `r`; writes to `x0` are discarded.
    fn set_reg(&mut self, r: u32, v: Value) {
        let r = (r & 31) as usize;
        if r == 0 {
            return;
        }
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        {
            let var = self.regs[r];
            self.b.def_var(var, v);
            // Checks made through the old value no longer apply.
            self.checked[0][r] = None;
            self.checked[1][r] = None;
        }
    }

    /// FP register `r` as raw bits.
    fn freg(&mut self, r: u32) -> Value {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        let var = self.fregs[(r & 31) as usize];
        self.b.use_var(var)
    }

    fn set_freg(&mut self, r: u32, v: Value) {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        let var = self.fregs[(r & 31) as usize];
        self.b.def_var(var, v);
    }

    /// FP register `r` as an `F32` (its low half) or `F64`.
    fn fval(&mut self, r: u32, dbl: bool) -> Value {
        let bits = self.freg(r);
        if dbl {
            self.b.ins().bitcast(F64, MemFlagsData::new(), bits)
        } else {
            let lo = self.b.ins().ireduce(I32, bits);
            self.b.ins().bitcast(F32, MemFlagsData::new(), lo)
        }
    }

    /// Write `v` (`F32` or `F64`) to FP register `rd`. A single goes into the
    /// low half, keeping the upper one (no NaN-boxing, as the interpreter).
    fn set_fval(&mut self, rd: u32, v: Value) {
        if self.b.func.dfg.value_type(v) == F64 {
            let bits = self.b.ins().bitcast(I64, MemFlagsData::new(), v);
            self.set_freg(rd, bits);
        } else {
            let bits = self.b.ins().bitcast(I32, MemFlagsData::new(), v);
            self.set_low_word(rd, bits);
        }
    }

    /// Replace the low 32 bits of FP register `rd` with the `I32` `word`.
    fn set_low_word(&mut self, rd: u32, word: Value) {
        let word = self.b.ins().uextend(I64, word);
        let old = self.freg(rd);
        let upper = self
            .b
            .ins()
            .band_imm_s(old, 0xFFFF_FFFF_0000_0000u64 as i64);
        let v = self.b.ins().bor(upper, word);
        self.set_freg(rd, v);
    }

    fn count(&mut self) -> Value {
        self.b.use_var(self.counter)
    }

    fn body(&mut self, insns: &[usize]) -> Option<()> {
        // `open`: the current block can still fall through.
        let mut open = false;
        for (pos, &i) in insns.iter().enumerate() {
            let ins = *self.insn(i)?;
            if let Some(&label) = self.labels.get(&ins.pc) {
                if open {
                    // Sequential flow into an entry point: no new block.
                    self.b.ins().jump(label, &[]);
                }
                self.b.switch_to_block(label);
                // Other paths reach the label without this run's checks.
                self.checked = [[None; 32]; 2];
            } else if !open {
                // Unreachable by construction: a non-entry instruction is
                // only ever reached by falling through.
                return None;
            }
            self.pos = pos;
            self.index = i;
            open = self.instruction(&ins)?;
            if open && ins.funcblock {
                // C++ `RV32I_BC_FUNCBLOCK`: `NEXT_BLOCK` with the overflow
                // check. (It also leaves a stale segment, but nothing native
                // code runs can make one stale.)
                let next = ins.pc.wrapping_add(ins.len);
                self.check_counter(next);
                self.enter_block(next);
                open = false;
            } else if open {
                let next = ins.pc.wrapping_add(ins.len);
                if self.a.region_at(self.exec, next) != Some(self.region) {
                    self.leave_mid_block(next);
                    open = false;
                }
            }
        }
        if open {
            return None;
        }
        Some(())
    }

    /// Continue at `target` as the start of a block (C++ `BEGIN_BLOCK`).
    fn enter_block(&mut self, target: u64) {
        let c = self.count();
        match self.a.region_at(self.exec, target) {
            Some(r) if r == self.region => {
                let label = self.labels.get(&target).copied();
                if let Some(label) = label {
                    let k = self.block_count(target);
                    let c = self.b.ins().iadd_imm_s(c, k as i64);
                    self.b.def_var(self.counter, c);
                    self.b.ins().jump(label, &[]);
                } else {
                    // Every in-region target is an entry; be safe anyway.
                    self.leave(EXIT_JUMP, target, c);
                }
            }
            Some(r) => self.tail_call(r, target, c),
            None => self.leave(EXIT_JUMP, target, c),
        }
    }

    /// Continue at `target` in the middle of a block: whoever runs it adds
    /// the block count for `target` again, so subtract it first.
    fn leave_mid_block(&mut self, target: u64) {
        let c = self.count();
        let k = self.block_count(target);
        let c = self.b.ins().iadd_imm_s(c, (k as i64).wrapping_neg());
        match self.a.region_at(self.exec, target) {
            Some(r) => self.tail_call(r, target, c),
            None => self.leave(EXIT_JUMP, target, c),
        }
    }

    /// The instruction count beginning a block at `target` adds: none in
    /// code that ignores the limit.
    fn block_count(&self, target: u64) -> u64 {
        if self.layout.unlimited {
            0
        } else {
            block_icount(self.exec, target)
        }
    }

    /// `if counter >= max { halt at target }`, as the dispatcher checks
    /// before jumping. Nothing in code that ignores the limit.
    fn check_counter(&mut self, target: u64) {
        if self.layout.unlimited {
            return;
        }
        let over = self.over_limit();
        self.exit_if(over, EXIT_HALT, ExitPc::Const(target));
    }

    /// `counter >= max`. The limit is read from the state block at every
    /// check (as C++ asmjit compares against memory) rather than kept in a
    /// register for the whole function.
    fn over_limit(&mut self) -> Value {
        let c = self.count();
        let max = self
            .b
            .ins()
            .load(I64, host_flags(), self.state, JitState::OFFSET_MAX);
        self.b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, c, max)
    }

    /// Leave with `kind` at `pc` (and the current counter) if `cond`. The
    /// exit is a cold block of its own, so its arguments are only
    /// materialised on the way out, not on the fast path.
    fn exit_if(&mut self, cond: Value, kind: u64, pc: ExitPc) {
        let c = self.count();
        let cold = self.b.create_block();
        self.b.set_cold_block(cold);
        let cont = self.b.create_block();
        self.b.ins().brif(cond, cold, &[], cont, &[]);
        self.b.switch_to_block(cold);
        let kind = self.b.ins().iconst(I64, kind as i64);
        let pc = match pc {
            ExitPc::Const(pc) => self.b.ins().iconst(I64, pc as i64),
            ExitPc::Value(pc) => pc,
        };
        self.b.ins().jump(self.exit, &args(&[kind, pc, c]));
        self.b.switch_to_block(cont);
    }

    fn leave(&mut self, kind: u64, pc: u64, counter: Value) {
        let kind = self.b.ins().iconst(I64, kind as i64);
        let pc = self.b.ins().iconst(I64, pc as i64);
        self.b.ins().jump(self.exit, &args(&[kind, pc, counter]));
    }

    fn tail_call(&mut self, region: RegionId, pc: u64, counter: Value) {
        let blk = *self.xfer.entry(region).or_insert_with(|| {
            let blk = self.b.create_block();
            self.b.append_block_param(blk, I64);
            self.b.append_block_param(blk, I64);
            blk
        });
        let pc = self.b.ins().iconst(I64, pc as i64);
        self.b.ins().jump(blk, &args(&[pc, counter]));
    }

    fn store_regs(&mut self) {
        for n in 0..32u32 {
            if self.written.get(n as usize).copied().unwrap_or(false) {
                let v = self.reg(n);
                self.b
                    .ins()
                    .store(host_flags(), v, self.regs_ptr, (n * 8) as i32);
            }
            if self.fwritten.get(n as usize).copied().unwrap_or(false) {
                let v = self.freg(n);
                self.b
                    .ins()
                    .store(host_flags(), v, self.fregs_ptr, (n * 8) as i32);
            }
        }
    }

    /// The shared exit and tail-call blocks.
    fn emit_exits(&mut self) {
        // exit: store registers, then exit_stored.
        self.b.switch_to_block(self.exit);
        let p = self.b.block_params(self.exit).to_vec();
        self.store_regs();
        self.b.ins().jump(self.exit_stored, &args(&p));

        self.b.switch_to_block(self.exit_stored);
        if let &[kind, pc, counter] = self.b.block_params(self.exit_stored) {
            let st = self.state;
            self.b
                .ins()
                .store(host_flags(), kind, st, JitState::OFFSET_EXIT_KIND);
            self.b
                .ins()
                .store(host_flags(), pc, st, JitState::OFFSET_EXIT_PC);
            self.b
                .ins()
                .store(host_flags(), counter, st, JitState::OFFSET_COUNTER);
            let done = self.b.ins().iconst(I64, DONE as i64);
            self.b.ins().return_(&[done, counter]);
        }

        let mut xfer: Vec<_> = self.xfer.iter().map(|(&r, &b)| (r, b)).collect();
        xfer.sort_unstable();
        for (region, blk) in xfer {
            self.b.switch_to_block(blk);
            let p = self.b.block_params(blk).to_vec();
            self.store_regs();
            let func = self.func_ref(region);
            if let &[pc, counter] = &p[..] {
                self.b
                    .ins()
                    .return_call(func, &[self.state, pc, counter, self.depth]);
            }
        }

        if let Some(blk) = self.ret {
            self.b.switch_to_block(blk);
            let p = self.b.block_params(blk).to_vec();
            self.store_regs();
            self.b.ins().return_(&p);
        }

        if let Some(blk) = self.indirect {
            self.b.switch_to_block(blk);
            let p = self.b.block_params(blk).to_vec();
            let &[target, counter] = &p[..] else { return };
            self.store_regs();
            // slot = (target >> SHIFT) - (begin >> SHIFT), as `DecodedSegment::index`
            let shifted = self.b.ins().ushr_imm_s(target, i64::from(SHIFT));
            let begin = (self.exec.exec_begin() >> SHIFT) as i64;
            let slot = self.b.ins().iadd_imm_s(shifted, begin.wrapping_neg());
            let st = self.state;
            let len = self
                .b
                .ins()
                .load(I64, host_flags(), st, JitState::OFFSET_TABLE_LEN);
            let in_table = self.b.ins().icmp(IntCC::UnsignedLessThan, slot, len);
            let lookup = self.b.create_block();
            let miss = self.b.create_block();
            self.b.ins().brif(in_table, lookup, &[], miss, &[]);

            self.b.switch_to_block(lookup);
            let table = self
                .b
                .ins()
                .load(I64, host_flags(), st, JitState::OFFSET_TABLE);
            let byte_off = self.b.ins().ishl_imm_s(slot, 3);
            let addr = self.b.ins().iadd(table, byte_off);
            let func = self.b.ins().load(I64, host_flags(), addr, 0);
            let call = self.b.create_block();
            self.b.ins().brif(func, call, &[], miss, &[]);

            self.b.switch_to_block(call);
            self.b.ins().return_call_indirect(
                self.tail_sig,
                func,
                &[st, target, counter, self.depth],
            );

            self.b.switch_to_block(miss);
            let kind = self.b.ins().iconst(I64, EXIT_INDIRECT as i64);
            self.b
                .ins()
                .jump(self.exit_stored, &args(&[kind, target, counter]));
        }
    }

    fn func_ref(&mut self, region: RegionId) -> FuncRef {
        if let Some(&f) = self.funcs.get(&region) {
            return f;
        }
        let name = self
            .b
            .func
            .declare_imported_user_function(UserExternalName::new(0, region as u32));
        let f = self.b.import_function(ExtFuncData {
            name: ExternalName::user(name),
            signature: self.tail_sig,
            colocated: true,
            patchable: false,
        });
        self.funcs.insert(region, f);
        f
    }

    /// The host address of the access at `rs1 + imm`, bounds-checked
    /// against the load or store window `[base, base + len)`. Like
    /// `Memory`'s fast path, anything else goes to the page table: the
    /// interpreter takes over at the instruction at `pc`. With an N-bit
    /// address space, mask it instead, without a check.
    ///
    /// The check covers the accesses through `rs1` later in the run as well
    /// (see [`Self::check_span`] and the module documentation); one that an
    /// earlier check covers has none of its own.
    fn guest_addr(&mut self, rs1: u32, imm: i64, write: bool, pc: u64) -> Value {
        let base = self.reg(rs1);
        let addr = self.b.ins().iadd_imm_s(base, imm);
        let mask = self.layout.mask;
        if mask != 0 {
            let index = if mask == u64::from(u32::MAX) {
                // A 32-bit zero extension, without materialising the mask.
                let low = self.b.ins().ireduce(I32, addr);
                self.b.ins().uextend(I64, low)
            } else {
                self.b.ins().band_imm_s(addr, mask as i64)
            };
            return self.b.ins().iadd(self.arena, index);
        }
        let reg = (rs1 & 31) as usize;
        let within = |span: Option<Span>| span.is_some_and(|(lo, hi)| lo <= imm && imm <= hi);
        let covered_by = |kind: usize| {
            within(
                self.checked
                    .get(kind)
                    .and_then(|k| k.get(reg))
                    .copied()
                    .flatten(),
            )
        };
        // A store check also covers loads when every writable address is
        // readable.
        let stores_cover_loads = window_within(self.layout.write, self.layout.read);
        let covered = if write {
            covered_by(1)
        } else {
            covered_by(0) || (stores_cover_loads && covered_by(1))
        };
        if !covered {
            let (loads, stores) = self.check_spans(rs1);
            // A load inside the span of the stores that follow it is checked
            // with them: it lies between two of their addresses, so it is
            // inside the window when they are, and the check only fails
            // when one of the stores would leave native code anyway.
            let as_store = write || (stores_cover_loads && within(stores));
            let (lo, hi) = if as_store { stores } else { loads }.unwrap_or((imm, imm));
            let (win_base, win_len) = if as_store {
                self.layout.write
            } else {
                self.layout.read
            };
            // Every `rs1 + k` for k in lo..=hi is inside the window exactly
            // when `rs1 + lo` is inside its first `len - (hi - lo)` bytes:
            // the span is below 2^13 and the window below 2^64, so nothing
            // wraps in between.
            let span = hi.wrapping_sub(lo) as u64;
            let first = self
                .b
                .ins()
                .iadd_imm_s(base, lo.wrapping_sub(win_base as i64));
            let outside = self.b.ins().icmp_imm_s(
                IntCC::UnsignedGreaterThanOrEqual,
                first,
                win_len.saturating_sub(span) as i64,
            );
            self.exit_if(outside, EXIT_SLOW, ExitPc::Const(pc));
            if let Some(slot) = self
                .checked
                .get_mut(usize::from(as_store))
                .and_then(|k| k.get_mut(reg))
            {
                *slot = Some((lo, hi));
            }
        }
        self.b.ins().iadd(self.arena, addr)
    }

    /// The offset spans `(lo, hi)` of the loads and of the stores through
    /// `rs1` from the instruction at `self.pos` to the end of its
    /// straight-line run (the next label, or an instruction that doesn't
    /// fall through), up to and including the first instruction that writes
    /// `rs1`: what a check made there can cover.
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
            let Some(ins) = self.insn(i) else { break };
            // Straight-line code: the next instruction, reached only by
            // falling through, not a label (which other paths reach).
            if let Some(p) = prev
                && (p.op.is_terminator()
                    || p.funcblock
                    || p.pc.wrapping_add(p.len) != ins.pc
                    || self.labels.contains_key(&ins.pc))
            {
                break;
            }
            if let Some((base, off, write)) = access(ins.op)
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
        match ins.op {
            JOp::Nop => {}
            JOp::Dyncall => {
                // Leave with every register stored and the block counted;
                // the dispatcher runs the handler and re-enters at `next`.
                let c = self.count();
                self.leave(EXIT_DYNCALL, ins.pc, c);
                return Some(false);
            }
            JOp::Fence => {
                self.b.ins().fence();
            }
            JOp::Const { rd, value } => {
                let v = self.b.ins().iconst(I64, value as i64);
                self.set_reg(rd, v);
            }
            JOp::AluImm { op, rd, rs1, imm } => {
                let a = self.reg(rs1);
                let v = self.alu_imm(op, a, imm);
                self.set_reg(rd, v);
            }
            JOp::Alu { op, rd, rs1, rs2 } => {
                let a = self.reg(rs1);
                let b = self.reg(rs2);
                let v = self.alu(op, a, b);
                self.set_reg(rd, v);
            }
            JOp::Unary { op, rd, rs1 } => {
                let a = self.reg(rs1);
                let v = self.unary(op, a);
                self.set_reg(rd, v);
            }
            JOp::Load {
                width,
                signed,
                rd,
                rs1,
                imm,
            } => {
                let host = self.guest_addr(rs1, imm, false, ins.pc);
                if let Some(rd) = rd {
                    let v = self.load(width, signed, host);
                    self.set_reg(rd, v);
                }
            }
            JOp::Store {
                width,
                rs1,
                rs2,
                imm,
            } => {
                let value = self.reg(rs2);
                let host = self.guest_addr(rs1, imm, true, ins.pc);
                self.store(width, value, host);
            }
            JOp::Branch {
                cond,
                rs1,
                rs2,
                target,
                check,
            } => {
                // Cranelift lowers a `brif`'s else successor right after
                // it. Forward branches keep the guest's fall-through there.
                // Backward branches (loops) invert the condition so the
                // taken path, the limit check and the jump back to the loop
                // head, comes next: one taken host branch per iteration
                // instead of two (C++ asmjit's `jcc notaken` + back-edge
                // check). Half the cycles of a tight loop.
                let backward = target <= ins.pc;
                let cc = if backward {
                    intcc(cond).complement()
                } else {
                    intcc(cond)
                };
                let a = self.reg(rs1);
                let test = match rs2 {
                    Some(rs2) => {
                        let b = self.reg(rs2);
                        self.b.ins().icmp(cc, a, b)
                    }
                    None => self.b.ins().icmp_imm_s(cc, a, 0),
                };
                let yes = self.b.create_block();
                let no = self.b.create_block();
                if backward {
                    self.b.ins().brif(test, no, &[], yes, &[]);
                } else {
                    self.b.ins().brif(test, yes, &[], no, &[]);
                }
                self.b.switch_to_block(yes);
                if check {
                    self.check_counter(target);
                }
                self.enter_block(target);
                self.b.switch_to_block(no);
                self.enter_block(next);
                return Some(false);
            }
            JOp::Jal { rd, target } => {
                if rd != 0 {
                    let link = self.b.ins().iconst(I64, next as i64);
                    self.set_reg(rd, link);
                }
                self.check_counter(target);
                match self.native_call(self.index) {
                    Some((_, callee)) => self.call(callee, target, next),
                    None => self.enter_block(target),
                }
                return Some(false);
            }
            // A constant target (`auipc`+`jalr`): like `jal`.
            JOp::Jalr { rd, .. }
                if let Some(target) = direct_jalr_target(self.exec, self.a, self.index) =>
            {
                if rd != 0 {
                    let link = self.b.ins().iconst(I64, next as i64);
                    self.set_reg(rd, link);
                }
                self.check_counter(target);
                match self.native_call(self.index) {
                    Some((_, callee)) => self.call(callee, target, next),
                    None => self.enter_block(target),
                }
                return Some(false);
            }
            JOp::Jalr { rd, rs1, imm } => {
                let base = self.reg(rs1);
                let address = self.b.ins().iadd_imm_s(base, imm);
                if rd != 0 {
                    let link = self.b.ins().iconst(I64, next as i64);
                    self.set_reg(rd, link);
                }
                let target = self.b.ins().band_imm_s(address, !1);
                if !COMPRESSED {
                    let low = self.b.ins().band_imm_s(target, 3);
                    self.exit_if(low, EXIT_MISALIGNED, ExitPc::Value(target));
                }
                // overflow_checked_jump
                if !self.layout.unlimited {
                    let over = self.over_limit();
                    self.exit_if(over, EXIT_HALT, ExitPc::Value(target));
                }
                let c = self.count();
                if is_return(ins.op) {
                    // To the native caller, which continues if `target` is
                    // its return address (or to the trampoline, which
                    // continues through the entry table).
                    let blk = self.ret_block();
                    self.b.ins().jump(blk, &args(&[target, c]));
                } else if rd != 0 && self.returns_here(next) {
                    self.indirect_call(target, next);
                } else {
                    let blk = self.indirect_block();
                    self.b.ins().jump(blk, &args(&[target, c]));
                }
                return Some(false);
            }
            JOp::FLoad {
                width,
                rd,
                rs1,
                imm,
            } => {
                let host = self.guest_addr(rs1, imm, false, ins.pc);
                if width == 4 {
                    let w = self.b.ins().load(I32, guest_flags(), host, 0);
                    self.set_low_word(rd, w);
                } else {
                    let v = self.b.ins().load(I64, guest_flags(), host, 0);
                    self.set_freg(rd, v);
                }
            }
            JOp::FStore {
                width,
                rs1,
                rs2,
                imm,
            } => {
                let value = self.freg(rs2);
                let host = self.guest_addr(rs1, imm, true, ins.pc);
                self.store(width, value, host);
            }
            JOp::FBin {
                op,
                dbl,
                rd,
                rs1,
                rs2,
            } => self.fbin(op, dbl, rd, rs1, rs2),
            JOp::FSqrt { dbl, rd, rs1 } => {
                let a = self.fval(rs1, dbl);
                let v = self.b.ins().sqrt(a);
                self.set_fval(rd, v);
            }
            JOp::FFma {
                op,
                dbl,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                let a = self.fval(rs1, dbl);
                let b = self.fval(rs2, dbl);
                let c = self.fval(rs3, dbl);
                let ins = &mut self.b;
                let v = match op {
                    FFma::Madd => ins.ins().fma(a, b, c),
                    FFma::Msub => {
                        let c = ins.ins().fneg(c);
                        ins.ins().fma(a, b, c)
                    }
                    FFma::Nmsub => {
                        let a = ins.ins().fneg(a);
                        ins.ins().fma(a, b, c)
                    }
                    FFma::Nmadd => {
                        let r = ins.ins().fma(a, b, c);
                        ins.ins().fneg(r)
                    }
                };
                self.set_fval(rd, v);
            }
            JOp::FCvtFF { to_dbl, rd, rs1 } => {
                let a = self.fval(rs1, !to_dbl);
                // A NaN converts to the canonical NaN (the interpreter's
                // `FCvtFF`); the host's conversion keeps sign and payload.
                let ins = &mut self.b;
                let (v, canonical) = if to_dbl {
                    let c = ins.ins().f64const(Ieee64::with_bits(0x7FF8_0000_0000_0000));
                    (ins.ins().fpromote(F64, a), c)
                } else {
                    let c = ins.ins().f32const(Ieee32::with_bits(0x7FC0_0000));
                    (ins.ins().fdemote(F32, a), c)
                };
                let nan = ins.ins().fcmp(FloatCC::Unordered, a, a);
                let v = ins.ins().select(nan, canonical, v);
                self.set_fval(rd, v);
            }
            JOp::FCmp {
                op,
                dbl,
                rd,
                rs1,
                rs2,
            } => {
                let a = self.fval(rs1, dbl);
                let b = self.fval(rs2, dbl);
                let cc = match op {
                    FCmp::Eq => FloatCC::Equal,
                    FCmp::Lt => FloatCC::LessThan,
                    FCmp::Le => FloatCC::LessThanOrEqual,
                };
                let c = self.b.ins().fcmp(cc, a, b);
                let v = self.b.ins().uextend(I64, c);
                self.set_reg(rd, v);
            }
            JOp::FToInt {
                kind,
                dbl,
                rm,
                rd,
                rs1,
            } => {
                let a = self.fval(rs1, dbl);
                let v = self.fcvt_to_int(kind, rm, a);
                self.set_reg(rd, v);
            }
            JOp::IntToF { kind, dbl, rd, rs1 } => {
                let x = self.reg(rs1);
                let ty = if dbl { F64 } else { F32 };
                let ins = &mut self.b;
                let v = match kind {
                    FInt::W => {
                        let w = ins.ins().ireduce(I32, x);
                        ins.ins().fcvt_from_sint(ty, w)
                    }
                    FInt::Wu => {
                        let w = ins.ins().ireduce(I32, x);
                        ins.ins().fcvt_from_uint(ty, w)
                    }
                    FInt::L => ins.ins().fcvt_from_sint(ty, x),
                    FInt::Lu => ins.ins().fcvt_from_uint(ty, x),
                };
                self.set_fval(rd, v);
            }
            JOp::FMvXF { dbl, rd, rs1 } => {
                let bits = self.freg(rs1);
                let v = if dbl {
                    bits
                } else {
                    let lo = self.b.ins().ireduce(I32, bits);
                    self.b.ins().sextend(I64, lo)
                };
                self.set_reg(rd, v);
            }
            JOp::FClass { dbl, rd, rs1 } => {
                let bits = self.freg(rs1);
                let v = self.fclass(dbl, bits);
                self.set_reg(rd, v);
            }
            JOp::FMvFX { dbl, rd, rs1 } => {
                let x = self.reg(rs1);
                if dbl {
                    self.set_freg(rd, x);
                } else {
                    let lo = self.b.ins().ireduce(I32, x);
                    self.set_low_word(rd, lo);
                }
            }
        }
        Some(true)
    }

    fn fbin(&mut self, op: FBin, dbl: bool, rd: u32, rs1: u32, rs2: u32) {
        if matches!(op, FBin::SgnJ | FBin::SgnJN | FBin::SgnJX) {
            // Bit operations on the raw patterns; a single's are in the low
            // half.
            let a = self.freg(rs1);
            let b = self.freg(rs2);
            let sign: i64 = if dbl { i64::MIN } else { 0x8000_0000 };
            let ins = &mut self.b;
            let s = match op {
                FBin::SgnJ => ins.ins().band_imm_s(b, sign),
                FBin::SgnJN => {
                    let n = ins.ins().bnot(b);
                    ins.ins().band_imm_s(n, sign)
                }
                _ => {
                    let x = ins.ins().bxor(a, b);
                    ins.ins().band_imm_s(x, sign)
                }
            };
            let mag = ins.ins().band_imm_s(a, !sign);
            let v = ins.ins().bor(mag, s);
            if dbl {
                self.set_freg(rd, v);
            } else {
                let lo = self.b.ins().ireduce(I32, v);
                self.set_low_word(rd, lo);
            }
            return;
        }
        let a = self.fval(rs1, dbl);
        let b = self.fval(rs2, dbl);
        let ins = &mut self.b;
        let v = match op {
            FBin::Add => ins.ins().fadd(a, b),
            FBin::Sub => ins.ins().fsub(a, b),
            FBin::Mul => ins.ins().fmul(a, b),
            FBin::Div => ins.ins().fdiv(a, b),
            _ => {
                // Cranelift's fmin/fmax order -0 below +0 like RISC-V, but
                // propagate NaN. RISC-V (and the interpreter's `rv_fmin32`
                // & co): one NaN operand gives the other operand (IEEE
                // minNum, C's fmin), two give the canonical NaN. Both are
                // defined results, not NaN details left to the host.
                let m = if op == FBin::Min {
                    ins.ins().fmin(a, b)
                } else {
                    ins.ins().fmax(a, b)
                };
                let a_nan = ins.ins().fcmp(FloatCC::Unordered, a, a);
                let b_nan = ins.ins().fcmp(FloatCC::Unordered, b, b);
                let both_nan = ins.ins().band(a_nan, b_nan);
                // `rvfd::CANONICAL_NAN_F64`/`_F32` (that module needs `ext-f`).
                let canonical = if dbl {
                    ins.ins().f64const(Ieee64::with_bits(0x7FF8_0000_0000_0000))
                } else {
                    ins.ins().f32const(Ieee32::with_bits(0x7FC0_0000))
                };
                let m = ins.ins().select(b_nan, a, m);
                let m = ins.ins().select(a_nan, b, m);
                ins.ins().select(both_nan, canonical, m)
            }
        };
        self.set_fval(rd, v);
    }

    /// `fcvt_round`: round `x` per the RISC-V rounding mode `rm` (a
    /// constant). Reserved modes round to nearest-even.
    fn round(&mut self, rm: u32, x: Value) -> Value {
        let ins = &mut self.b;
        match rm {
            1 => ins.ins().trunc(x),
            2 => ins.ins().floor(x),
            3 => ins.ins().ceil(x),
            4 => {
                // Ties away from zero: x - trunc(x) is exact.
                let ty = ins.func.dfg.value_type(x);
                let t = ins.ins().trunc(x);
                let frac = ins.ins().fsub(x, t);
                let frac = ins.ins().fabs(frac);
                let (half, one) = if ty == F64 {
                    (ins.ins().f64const(0.5), ins.ins().f64const(1.0))
                } else {
                    (ins.ins().f32const(0.5), ins.ins().f32const(1.0))
                };
                let away = ins.ins().fcmp(FloatCC::GreaterThanOrEqual, frac, half);
                let step = ins.ins().fcopysign(one, x);
                let up = ins.ins().fadd(t, step);
                ins.ins().select(away, up, t)
            }
            _ => ins.ins().nearest(x),
        }
    }

    /// `fcvt_to_integer` plus the interpreter's widening: round, convert
    /// with saturation, NaN to the maximum; 32-bit results sign-extended.
    fn fcvt_to_int(&mut self, kind: FInt, rm: u32, x: Value) -> Value {
        let rounded = if rm == 7 {
            // DYN: FCSR.frm, which native code can't change (CSR
            // instructions are interpreted).
            let frm = self
                .b
                .ins()
                .load(I64, host_flags(), self.state, JitState::OFFSET_FRM);
            let mut v = self.round(0, x);
            for mode in 1..=4 {
                let r = self.round(mode, x);
                let is = self.b.ins().icmp_imm_s(IntCC::Equal, frm, i64::from(mode));
                v = self.b.ins().select(is, r, v);
            }
            v
        } else if rm == 1 {
            // The saturating conversions truncate already.
            x
        } else {
            self.round(rm, x)
        };
        let ins = &mut self.b;
        let (v, max) = match kind {
            FInt::W => {
                let v = ins.ins().fcvt_to_sint_sat(I32, rounded);
                (ins.ins().sextend(I64, v), i64::from(i32::MAX))
            }
            FInt::Wu => {
                let v = ins.ins().fcvt_to_uint_sat(I32, rounded);
                (ins.ins().sextend(I64, v), -1)
            }
            FInt::L => (ins.ins().fcvt_to_sint_sat(I64, rounded), i64::MAX),
            FInt::Lu => (ins.ins().fcvt_to_uint_sat(I64, rounded), -1),
        };
        let nan = ins.ins().fcmp(FloatCC::Unordered, x, x);
        let max = ins.ins().iconst(I64, max);
        ins.ins().select(nan, max, v)
    }

    /// `rv_fclass32`/`rv_fclass64` on the raw bits: exactly one bit set.
    fn fclass(&mut self, dbl: bool, bits: Value) -> Value {
        let ins = &mut self.b;
        let (bits, exp_shift, exp_max, frac_mask, quiet) = if dbl {
            (bits, 52, 0x7FF, 0xF_FFFF_FFFF_FFFFi64, 1i64 << 51)
        } else {
            let lo = ins.ins().ireduce(I32, bits);
            (ins.ins().uextend(I64, lo), 23, 0xFF, 0x7F_FFFF, 1 << 22)
        };
        let sign_bit = if dbl { 63 } else { 31 };
        let sign = ins.ins().ushr_imm_s(bits, sign_bit);
        let negative = ins.ins().icmp_imm_s(IntCC::NotEqual, sign, 0);
        let exp = ins.ins().ushr_imm_s(bits, exp_shift);
        let exp = ins.ins().band_imm_s(exp, exp_max);
        let frac = ins.ins().band_imm_s(bits, frac_mask);
        let is_max = ins.ins().icmp_imm_s(IntCC::Equal, exp, exp_max);
        let is_zero_exp = ins.ins().icmp_imm_s(IntCC::Equal, exp, 0);
        let frac_zero = ins.ins().icmp_imm_s(IntCC::Equal, frac, 0);
        let q = ins.ins().band_imm_s(frac, quiet);
        let is_quiet = ins.ins().icmp_imm_s(IntCC::NotEqual, q, 0);
        let mut pick = |c: Value, yes: i64, no: i64| {
            let y = ins.ins().iconst(I64, yes);
            let n = ins.ins().iconst(I64, no);
            ins.ins().select(c, y, n)
        };
        let inf = pick(negative, 1 << 0, 1 << 7);
        let nan = pick(is_quiet, 1 << 9, 1 << 8);
        let zero = pick(negative, 1 << 3, 1 << 4);
        let subnormal = pick(negative, 1 << 2, 1 << 5);
        let normal = pick(negative, 1 << 1, 1 << 6);
        let special = ins.ins().select(frac_zero, inf, nan);
        let small = ins.ins().select(frac_zero, zero, subnormal);
        let v = ins.ins().select(is_zero_exp, small, normal);
        ins.ins().select(is_max, special, v)
    }

    /// The call at `a.insns[i]`, if it is made as a native call: `jal` or
    /// a constant `jalr` with a link register, to a region, returning to
    /// a label of this region. `(target, callee region)`.
    fn native_call(&self, i: usize) -> Option<(u64, RegionId)> {
        let ins = self.a.insns.get(i)?;
        let target = match ins.op {
            JOp::Jal { rd, target } if rd & 31 != 0 => target,
            JOp::Jalr { rd, .. } if rd & 31 != 0 => direct_jalr_target(self.exec, self.a, i)?,
            _ => return None,
        };
        let next = ins.pc.wrapping_add(ins.len);
        if !self.returns_here(next) {
            return None;
        }
        Some((target, self.a.region_at(self.exec, target)?))
    }

    /// A call returning to `next` can be native: `next` is a label here.
    fn returns_here(&self, next: u64) -> bool {
        self.a.region_at(self.exec, next) == Some(self.region) && self.labels.contains_key(&next)
    }

    /// `depth < DEPTH`: another native call fits.
    fn call_room(&mut self) -> Value {
        self.b
            .ins()
            .icmp_imm_s(IntCC::UnsignedLessThan, self.depth, i64::from(DEPTH))
    }

    /// A native call to `callee` at `target`, returning to `next` (C++
    /// asmjit `emit_direct_call`): store the registers the callee may
    /// read, call it, reload those it may write, and continue at `next` if
    /// it returned there. A return elsewhere goes on to this region's
    /// caller, and so does leaving for the host. Past `DEPTH` nested calls,
    /// it is a tail call like any jump.
    fn call(&mut self, callee: RegionId, target: u64, next: u64) {
        let Some(&Summary { reads, writes }) = self.summaries.get(callee) else {
            self.enter_block(target);
            return;
        };
        let room = self.call_room();
        let call = self.b.create_block();
        let deeper = self.b.create_block();
        self.b.set_cold_block(deeper);
        self.b.ins().brif(room, call, &[], deeper, &[]);
        self.b.switch_to_block(call);
        let func = self.func_ref(callee);
        let t = self.b.ins().iconst(I64, target as i64);
        self.native_call_body(Callee::Direct(func), t, reads, writes, next);
        self.b.switch_to_block(deeper);
        self.enter_block(target);
    }

    /// An indirect call to `target` (a register's value) returning to
    /// `next`: a native call through the entry table, which may read and
    /// write every register; otherwise (no entry, too deep) the indirect
    /// jump it is.
    fn indirect_call(&mut self, target: Value, next: u64) {
        let c = self.count();
        let jump = self.indirect_block();
        let room = self.call_room();
        let lookup = self.b.create_block();
        self.b
            .ins()
            .brif(room, lookup, &[], jump, &args(&[target, c]));
        self.b.switch_to_block(lookup);
        // slot = (target >> SHIFT) - (begin >> SHIFT), as `DecodedSegment::index`
        let shifted = self.b.ins().ushr_imm_s(target, i64::from(SHIFT));
        let begin = (self.exec.exec_begin() >> SHIFT) as i64;
        let slot = self.b.ins().iadd_imm_s(shifted, begin.wrapping_neg());
        let st = self.state;
        let len = self
            .b
            .ins()
            .load(I64, host_flags(), st, JitState::OFFSET_TABLE_LEN);
        let in_table = self.b.ins().icmp(IntCC::UnsignedLessThan, slot, len);
        let load = self.b.create_block();
        self.b
            .ins()
            .brif(in_table, load, &[], jump, &args(&[target, c]));
        self.b.switch_to_block(load);
        let table = self
            .b
            .ins()
            .load(I64, host_flags(), st, JitState::OFFSET_TABLE);
        let byte_off = self.b.ins().ishl_imm_s(slot, 3);
        let addr = self.b.ins().iadd(table, byte_off);
        let func = self.b.ins().load(I64, host_flags(), addr, 0);
        let call = self.b.create_block();
        self.b
            .ins()
            .brif(func, call, &[], jump, &args(&[target, c]));
        self.b.switch_to_block(call);
        self.native_call_body(Callee::Indirect(func), target, !0, !0, next);
    }

    /// The call itself, for a callee that may read `reads` and write
    /// `writes`: on return to `next`, enters `next`.
    fn native_call_body(
        &mut self,
        callee: Callee,
        target: Value,
        reads: RegSet,
        writes: RegSet,
        next: u64,
    ) {
        let (store, load) = (self.written_set & reads, self.used_set & writes);
        for n in 1..64u32 {
            if store >> n & 1 == 0 {
                continue;
            }
            let (v, ptr) = if n < 32 {
                (self.reg(n), self.regs_ptr)
            } else {
                (self.freg(n - 32), self.fregs_ptr)
            };
            self.b
                .ins()
                .store(host_flags(), v, ptr, ((n & 31) * 8) as i32);
        }
        let c = self.count();
        let depth = self.b.ins().iadd_imm_s(self.depth, 1);
        let call_args = [self.state, target, c, depth];
        let call = match callee {
            Callee::Direct(func) => self.b.ins().call(func, &call_args),
            Callee::Indirect(func) => self.b.ins().call_indirect(self.tail_sig, func, &call_args),
        };
        let results = self.b.inst_results(call).to_vec();
        let &[ct, cc] = &results[..] else { return };
        for n in 1..64u32 {
            if load >> n & 1 == 0 {
                continue;
            }
            let ptr = if n < 32 {
                self.regs_ptr
            } else {
                self.fregs_ptr
            };
            let v = self
                .b
                .ins()
                .load(I64, host_flags(), ptr, ((n & 31) * 8) as i32);
            // The index is masked to 0..32, so it is always in bounds.
            #[allow(clippy::indexing_slicing)]
            let var = if n < 32 {
                self.regs[(n & 31) as usize]
            } else {
                self.fregs[(n & 31) as usize]
            };
            self.b.def_var(var, v);
        }
        self.b.def_var(self.counter, cc);
        let at_next = self.b.ins().icmp_imm_s(IntCC::Equal, ct, next as i64);
        let cont = self.b.create_block();
        let ret = self.ret_block();
        self.b.ins().brif(at_next, cont, &[], ret, &args(&[ct, cc]));
        self.b.switch_to_block(cont);
        self.enter_block(next);
    }

    fn ret_block(&mut self) -> Block {
        if let Some(b) = self.ret {
            return b;
        }
        let blk = self.b.create_block();
        self.b.append_block_param(blk, I64);
        self.b.append_block_param(blk, I64);
        self.ret = Some(blk);
        blk
    }

    fn indirect_block(&mut self) -> Block {
        if let Some(b) = self.indirect {
            return b;
        }
        let blk = self.b.create_block();
        self.b.append_block_param(blk, I64);
        self.b.append_block_param(blk, I64);
        self.indirect = Some(blk);
        blk
    }

    fn load(&mut self, width: u8, signed: bool, host: Value) -> Value {
        let f = guest_flags();
        let ins = self.b.ins();
        match (width, signed) {
            (1, true) => ins.sload8(I64, f, host, 0),
            (1, false) => ins.uload8(I64, f, host, 0),
            (2, true) => ins.sload16(I64, f, host, 0),
            (2, false) => ins.uload16(I64, f, host, 0),
            (4, true) => ins.sload32(f, host, 0),
            (4, false) => ins.uload32(f, host, 0),
            _ => ins.load(I64, f, host, Offset32::new(0)),
        }
    }

    fn store(&mut self, width: u8, value: Value, host: Value) {
        let f = guest_flags();
        let ins = self.b.ins();
        match width {
            1 => ins.istore8(f, value, host, 0),
            2 => ins.istore16(f, value, host, 0),
            4 => ins.istore32(f, value, host, 0),
            _ => ins.store(f, value, host, 0),
        };
    }

    /// Apply a 32-bit operation and sign-extend the result (RV64 `*W`).
    fn word(
        &mut self,
        a: Value,
        f: impl FnOnce(&mut FunctionBuilder<'f>, Value) -> Value,
    ) -> Value {
        let a32 = self.b.ins().ireduce(I32, a);
        let r = f(&mut self.b, a32);
        self.b.ins().sextend(I64, r)
    }

    fn alu_imm(&mut self, op: AluI, a: Value, imm: i64) -> Value {
        match op {
            AluI::Add => self.b.ins().iadd_imm_s(a, imm),
            AluI::Slt => {
                let c = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, a, imm);
                self.b.ins().uextend(I64, c)
            }
            AluI::Sltu => {
                let c = self.b.ins().icmp_imm_s(IntCC::UnsignedLessThan, a, imm);
                self.b.ins().uextend(I64, c)
            }
            AluI::Xor => self.b.ins().bxor_imm_s(a, imm),
            AluI::Or => self.b.ins().bor_imm_s(a, imm),
            AluI::And => self.b.ins().band_imm_s(a, imm),
            AluI::Sll => self.b.ins().ishl_imm_s(a, imm),
            AluI::Srl => self.b.ins().ushr_imm_s(a, imm),
            AluI::Sra => self.b.ins().sshr_imm_s(a, imm),
            AluI::AddW => self.word(a, |b, x| b.ins().iadd_imm_s(x, imm)),
            AluI::SllW => self.word(a, |b, x| b.ins().ishl_imm_s(x, imm)),
            AluI::SrlW => self.word(a, |b, x| b.ins().ushr_imm_s(x, imm)),
            AluI::SraW => self.word(a, |b, x| b.ins().sshr_imm_s(x, imm)),
            AluI::Ror => self.b.ins().rotr_imm_s(a, imm),
            AluI::RorW => self.word(a, |b, x| b.ins().rotr_imm_s(x, imm)),
            AluI::Bset => self.b.ins().bor_imm_s(a, 1i64.wrapping_shl(imm as u32)),
            AluI::Bclr => self.b.ins().band_imm_s(a, !1i64.wrapping_shl(imm as u32)),
            AluI::Binv => self.b.ins().bxor_imm_s(a, 1i64.wrapping_shl(imm as u32)),
            AluI::Bext => {
                let v = self.b.ins().ushr_imm_s(a, imm);
                self.b.ins().band_imm_s(v, 1)
            }
            AluI::SlliUw => {
                let lo = self.b.ins().ireduce(I32, a);
                let lo = self.b.ins().uextend(I64, lo);
                self.b.ins().ishl_imm_s(lo, imm)
            }
        }
    }

    fn unary(&mut self, op: Unary, a: Value) -> Value {
        let ext = |b: &mut FunctionBuilder<'_>, ty: Type, signed: bool| {
            let n = b.ins().ireduce(ty, a);
            if signed {
                b.ins().sextend(I64, n)
            } else {
                b.ins().uextend(I64, n)
            }
        };
        match op {
            Unary::ZextB => ext(&mut self.b, I8, false),
            Unary::SextB => ext(&mut self.b, I8, true),
            Unary::ZextH => ext(&mut self.b, I16, false),
            Unary::SextH => ext(&mut self.b, I16, true),
            Unary::ZextW => ext(&mut self.b, I32, false),
            Unary::Not => self.b.ins().bnot(a),
            // Cranelift defines clz/ctz of zero as the bit width, like RISC-V.
            Unary::Clz => self.b.ins().clz(a),
            Unary::Ctz => self.b.ins().ctz(a),
            Unary::Cpop => self.b.ins().popcnt(a),
            Unary::ClzW | Unary::CtzW | Unary::CpopW => {
                let w = self.b.ins().ireduce(I32, a);
                let r = match op {
                    Unary::ClzW => self.b.ins().clz(w),
                    Unary::CtzW => self.b.ins().ctz(w),
                    _ => self.b.ins().popcnt(w),
                };
                self.b.ins().uextend(I64, r)
            }
            Unary::Rev8 => self.b.ins().bswap(a),
            Unary::Brev8 => {
                let s = self.b.ins().bswap(a);
                self.b.ins().bitrev(s)
            }
            Unary::OrcB => {
                // A byte's top bit ends up set iff the byte is non-zero
                // (no carry crosses bytes); then spread it over the byte.
                const LOW7: i64 = 0x7F7F_7F7F_7F7F_7F7F;
                let ins = &mut self.b;
                let t = ins.ins().band_imm_s(a, LOW7);
                let t = ins.ins().iadd_imm_s(t, LOW7);
                let t = ins.ins().bor(t, a);
                let t = ins.ins().band_imm_s(t, 0x8080_8080_8080_8080u64 as i64);
                let t = ins.ins().ushr_imm_s(t, 7);
                ins.ins().imul_imm_s(t, 0xFF)
            }
        }
    }

    fn alu(&mut self, op: AluR, a: Value, b: Value) -> Value {
        let bx = &mut self.b;
        match op {
            AluR::Add => bx.ins().iadd(a, b),
            AluR::Sub => bx.ins().isub(a, b),
            // Cranelift masks shift amounts to the operand width, like RISC-V.
            AluR::Sll => bx.ins().ishl(a, b),
            AluR::Srl => bx.ins().ushr(a, b),
            AluR::Sra => bx.ins().sshr(a, b),
            AluR::Slt => {
                let c = bx.ins().icmp(IntCC::SignedLessThan, a, b);
                bx.ins().uextend(I64, c)
            }
            AluR::Sltu => {
                let c = bx.ins().icmp(IntCC::UnsignedLessThan, a, b);
                bx.ins().uextend(I64, c)
            }
            AluR::Xor => bx.ins().bxor(a, b),
            AluR::Or => bx.ins().bor(a, b),
            AluR::And => bx.ins().band(a, b),
            AluR::Mul => bx.ins().imul(a, b),
            AluR::Mulh => bx.ins().smulhi(a, b),
            AluR::Mulhu => bx.ins().umulhi(a, b),
            AluR::Mulhsu => {
                // mulhu(a, b) - (a < 0 ? b : 0)
                let hi = bx.ins().umulhi(a, b);
                let sign = bx.ins().sshr_imm_s(a, 63);
                let fix = bx.ins().band(sign, b);
                bx.ins().isub(hi, fix)
            }
            AluR::Div => signed_div(bx, I64, a, b, false),
            AluR::Rem => signed_div(bx, I64, a, b, true),
            AluR::Divu => unsigned_div(bx, a, b, false),
            AluR::Remu => unsigned_div(bx, a, b, true),
            AluR::AddW | AluR::SubW | AluR::SllW | AluR::SrlW | AluR::SraW | AluR::MulW => {
                let a = bx.ins().ireduce(I32, a);
                let b = bx.ins().ireduce(I32, b);
                let r = match op {
                    AluR::AddW => bx.ins().iadd(a, b),
                    AluR::SubW => bx.ins().isub(a, b),
                    AluR::SllW => bx.ins().ishl(a, b),
                    AluR::SrlW => bx.ins().ushr(a, b),
                    AluR::SraW => bx.ins().sshr(a, b),
                    _ => bx.ins().imul(a, b),
                };
                bx.ins().sextend(I64, r)
            }
            AluR::DivW | AluR::RemW => {
                let a = bx.ins().ireduce(I32, a);
                let b = bx.ins().ireduce(I32, b);
                let r = signed_div(bx, I32, a, b, op == AluR::RemW);
                bx.ins().sextend(I64, r)
            }
            AluR::ShAdd(n) => {
                let s = bx.ins().ishl_imm_s(a, i64::from(n));
                bx.ins().iadd(b, s)
            }
            AluR::ShAddUw(n) => {
                let lo = bx.ins().ireduce(I32, a);
                let lo = bx.ins().uextend(I64, lo);
                let s = bx.ins().ishl_imm_s(lo, i64::from(n));
                bx.ins().iadd(b, s)
            }
            AluR::Pack => {
                let lo = bx.ins().ireduce(I32, a);
                let lo = bx.ins().uextend(I64, lo);
                let hi = bx.ins().ishl_imm_s(b, 32);
                bx.ins().bor(lo, hi)
            }
            AluR::PackH => {
                let lo = bx.ins().band_imm_s(a, 0xFF);
                let hi = bx.ins().band_imm_s(b, 0xFF);
                let hi = bx.ins().ishl_imm_s(hi, 8);
                bx.ins().bor(lo, hi)
            }
            AluR::PackW => {
                let a = bx.ins().ireduce(I32, a);
                let b = bx.ins().ireduce(I32, b);
                let lo = bx.ins().band_imm_s(a, 0xFFFF);
                let hi = bx.ins().ishl_imm_s(b, 16);
                let r = bx.ins().bor(lo, hi);
                bx.ins().sextend(I64, r)
            }
            AluR::Min => bx.ins().smin(a, b),
            AluR::Max => bx.ins().smax(a, b),
            AluR::Minu => bx.ins().umin(a, b),
            AluR::Maxu => bx.ins().umax(a, b),
            AluR::CzeroEqz => {
                let z = bx.ins().iconst(I64, 0);
                bx.ins().select(b, a, z)
            }
            AluR::CzeroNez => {
                let z = bx.ins().iconst(I64, 0);
                bx.ins().select(b, z, a)
            }
            AluR::Bset | AluR::Bclr | AluR::Binv => {
                let one = bx.ins().iconst(I64, 1);
                let bit = bx.ins().ishl(one, b);
                match op {
                    AluR::Bset => bx.ins().bor(a, bit),
                    AluR::Bclr => bx.ins().band_not(a, bit),
                    _ => bx.ins().bxor(a, bit),
                }
            }
            AluR::Bext => {
                let v = bx.ins().ushr(a, b);
                bx.ins().band_imm_s(v, 1)
            }
            AluR::Xnor => bx.ins().bxor_not(a, b),
            AluR::Orn => bx.ins().bor_not(a, b),
            AluR::Andn => bx.ins().band_not(a, b),
            // Rotate amounts are taken modulo the width, like RISC-V.
            AluR::Rol => bx.ins().rotl(a, b),
            AluR::Ror => bx.ins().rotr(a, b),
            AluR::RolW | AluR::RorW => {
                let a = bx.ins().ireduce(I32, a);
                let b = bx.ins().ireduce(I32, b);
                let r = if op == AluR::RolW {
                    bx.ins().rotl(a, b)
                } else {
                    bx.ins().rotr(a, b)
                };
                bx.ins().sextend(I64, r)
            }
            AluR::DivuW | AluR::RemuW => {
                let a = bx.ins().ireduce(I32, a);
                let b = bx.ins().ireduce(I32, b);
                let r = unsigned_div(bx, a, b, op == AluR::RemuW);
                bx.ins().sextend(I64, r)
            }
        }
    }
}

/// RISC-V signed division/remainder, which never traps: `x / 0 = -1`,
/// `x % 0 = x`, `MIN / -1 = MIN`, `MIN % -1 = 0`.
fn signed_div(bx: &mut FunctionBuilder<'_>, ty: Type, a: Value, b: Value, rem: bool) -> Value {
    let zero = bx.ins().icmp_imm_s(IntCC::Equal, b, 0);
    let minus1 = bx.ins().icmp_imm_s(IntCC::Equal, b, -1);
    let special = bx.ins().bor(zero, minus1);
    let one = bx.ins().iconst(ty, 1);
    let safe = bx.ins().select(special, one, b);
    if rem {
        let r = bx.ins().srem(a, safe);
        let z = bx.ins().iconst(ty, 0);
        let r = bx.ins().select(minus1, z, r);
        bx.ins().select(zero, a, r)
    } else {
        let q = bx.ins().sdiv(a, safe);
        let neg = bx.ins().ineg(a);
        let q = bx.ins().select(minus1, neg, q);
        let all = bx.ins().iconst(ty, -1);
        bx.ins().select(zero, all, q)
    }
}

/// RISC-V unsigned division/remainder: `x / 0 = MAX`, `x % 0 = x`.
fn unsigned_div(bx: &mut FunctionBuilder<'_>, a: Value, b: Value, rem: bool) -> Value {
    let ty = bx.func.dfg.value_type(a);
    let zero = bx.ins().icmp_imm_s(IntCC::Equal, b, 0);
    let one = bx.ins().iconst(ty, 1);
    let safe = bx.ins().select(zero, one, b);
    if rem {
        let r = bx.ins().urem(a, safe);
        bx.ins().select(zero, a, r)
    } else {
        let q = bx.ins().udiv(a, safe);
        let all = bx.ins().iconst(ty, -1);
        bx.ins().select(zero, all, q)
    }
}

const fn intcc(cond: Cond) -> IntCC {
    match cond {
        Cond::Eq => IntCC::Equal,
        Cond::Ne => IntCC::NotEqual,
        Cond::Lt => IntCC::SignedLessThan,
        Cond::Ge => IntCC::SignedGreaterThanOrEqual,
        Cond::Ltu => IntCC::UnsignedLessThan,
        Cond::Geu => IntCC::UnsignedGreaterThanOrEqual,
    }
}

/// Offsets `lo..=hi` of guest memory accesses through one base register.
type Span = (i64, i64);

/// What a native call calls.
#[derive(Debug, Clone, Copy)]
enum Callee {
    Direct(FuncRef),
    /// A function pointer from the entry table.
    Indirect(Value),
}

/// The PC an exit leaves with.
#[derive(Debug, Clone, Copy)]
enum ExitPc {
    Const(u64),
    Value(Value),
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

/// The base register and offset of a guest memory access, and whether it
/// is a store.
const fn access(op: JOp) -> Option<(u32, i64, bool)> {
    match op {
        JOp::Load { rs1, imm, .. } | JOp::FLoad { rs1, imm, .. } => Some((rs1, imm, false)),
        JOp::Store { rs1, imm, .. } | JOp::FStore { rs1, imm, .. } => Some((rs1, imm, true)),
        _ => None,
    }
}
