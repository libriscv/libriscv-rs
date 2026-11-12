//! Port of `asmjit/aj_emit.cpp`, generating Cranelift IR instead of asmjit
//! code for the regions of one execute segment.
//!
//! Every region becomes one function with Cranelift's `tail` calling
//! convention and the signature `(state, pc, counter)`. It switches on `pc`
//! to the entry point, keeps the guest registers it uses in SSA variables,
//! and leaves in one of three ways:
//!
//! * a native tail call (`return_call`) to the region owning the next
//!   instruction, so guest calls, returns and loops never come back through
//!   the interpreter; indirect jumps go through the entry table
//!   (`return_call_indirect`);
//! * a return to the trampoline with an exit kind in the state block (the
//!   instruction limit, an instruction the JIT doesn't compile, a fault);
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
//! Generated code only touches the state block, the 32 integer registers
//! and the guest arena, and every arena access is preceded by the same range
//! check as `Memory::read_*`/`write_*`. `libriscv-jit-rt` validates the
//! checked windows against the real arena length. See its crate
//! documentation for the full contract.

use std::collections::HashMap;

use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::types::{I8, I16, I32, I64};
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, Endianness, ExtFuncData, ExternalName, FuncRef, Function,
    InstBuilder, MemFlagsData, SigRef, Signature, Type, UserExternalName, UserFuncName, Value,
};
use cranelift_codegen::isa::{CallConv, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{Context, RelocTarget};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use libriscv_jit_rt::{CodeMemory, JitState};

use super::classify::{AluI, AluR, Cond, JOp, Unary};
use super::region::{Analysis, Insn, RegionId, block_icount};
use super::{EXIT_FAULT, EXIT_HALT, EXIT_INDIRECT, EXIT_JUMP, EXIT_MISALIGNED};
use crate::decoder_cache::{COMPRESSED, DecodedSegment, SHIFT};

/// Functions are placed at this alignment in the code image.
const FUNCTION_ALIGN: usize = 64;

/// The compiled segment: code memory plus the decoder indices to patch.
pub(crate) struct Compiled {
    pub code: CodeMemory,
    pub patch: Vec<usize>,
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
    flags.set("is_pic", "false").ok()?;
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
    let isa = cranelift_native::builder().ok()?;
    isa.finish(settings::Flags::new(flags)).ok()
}

fn tail_signature() -> Signature {
    let mut sig = Signature::new(CallConv::Tail);
    for _ in 0..3 {
        sig.params.push(AbiParam::new(I64));
    }
    sig
}

/// One function's machine code and the calls in it that need linking.
struct Object {
    bytes: Vec<u8>,
    /// `(offset, kind, addend, target function)`
    relocs: Vec<(usize, Reloc, i64, usize)>,
}

fn compile_function(ctx: &mut Context, isa: &OwnedTargetIsa) -> Option<Object> {
    let mut ctrl = ControlPlane::default();
    let (bytes, raw) = {
        let code = ctx.compile(isa.as_ref(), &mut ctrl).ok()?;
        (code.code_buffer().to_vec(), code.buffer.relocs().to_vec())
    };
    let mut relocs = Vec::new();
    for r in &raw {
        let RelocTarget::ExternalName(ExternalName::User(name)) = &r.target else {
            return None;
        };
        let target = ctx.func.params.user_named_funcs().get(*name)?.index as usize;
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

/// Compile every region of `a`, link them and map the result executable.
pub(crate) fn compile(exec: &DecodedSegment, a: &Analysis) -> Option<Compiled> {
    if a.regions.is_empty() {
        return None;
    }
    let isa = host_isa()?;
    let mut ctx = Context::new();
    let mut fctx = FunctionBuilderContext::new();
    let mut objects = Vec::with_capacity(a.regions.len() + 1);
    for r in 0..a.regions.len() {
        ctx.clear();
        ctx.func = Function::with_name_signature(UserFuncName::user(0, r as u32), tail_signature());
        RegionEmitter::emit(exec, a, r, &mut ctx.func, &mut fctx, &isa)?;
        objects.push(compile_function(&mut ctx, &isa)?);
    }
    // The trampoline: called from Rust with the C calling convention as
    // `(state, function, pc, counter)`, it enters the `tail` world.
    ctx.clear();
    ctx.func = trampoline(&isa, &mut fctx);
    objects.push(compile_function(&mut ctx, &isa)?);

    // Link: lay out the functions, then resolve the direct calls.
    let mut image = Vec::new();
    let mut offsets = Vec::with_capacity(objects.len());
    for obj in &objects {
        image.resize(image.len().next_multiple_of(FUNCTION_ALIGN), 0);
        offsets.push(image.len());
        image.extend_from_slice(&obj.bytes);
    }
    for (obj, &base) in objects.iter().zip(&offsets) {
        for &(offset, kind, addend, target) in &obj.relocs {
            let what = i64::try_from(*offsets.get(target)?)
                .ok()?
                .checked_add(addend)?;
            relocate(&mut image, base + offset, kind, what)?;
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
    let code = CodeMemory::new(&image, trampoline, &entries).ok()?;
    let patch = a
        .block_starts
        .iter()
        .copied()
        .filter(|&i| code.has_entry(i))
        .collect();
    Some(Compiled { code, patch })
}

fn trampoline(isa: &OwnedTargetIsa, fctx: &mut FunctionBuilderContext) -> Function {
    let mut sig = Signature::new(isa.default_call_conv());
    for _ in 0..4 {
        sig.params.push(AbiParam::new(I64));
    }
    let mut func = Function::with_name_signature(UserFuncName::user(1, 0), sig);
    let mut b = FunctionBuilder::new(&mut func, fctx);
    let tail = b.import_signature(tail_signature());
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let p = b.block_params(entry).to_vec();
    if let [state, function, pc, counter] = p[..] {
        b.ins().call_indirect(tail, function, &[state, pc, counter]);
    }
    b.ins().return_(&[]);
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
    b: FunctionBuilder<'f>,
    regs: [Variable; 32],
    counter: Variable,
    /// Registers written anywhere in the region (stored on every exit).
    written: [bool; 32],
    state: Value,
    regs_ptr: Value,
    arena: Value,
    read_base: Value,
    read_len: Value,
    write_base: Value,
    write_len: Value,
    max: Value,
    labels: HashMap<u64, Block>,
    /// `(kind, pc, counter)`: store the registers, then leave.
    exit: Block,
    /// `(kind, pc, counter)`: leave; the registers are already stored.
    exit_stored: Block,
    /// Target region -> `(pc, counter)` block that tail-calls it.
    xfer: HashMap<RegionId, Block>,
    /// `(target, counter)`: an indirect jump through the entry table.
    indirect: Option<Block>,
    funcs: HashMap<RegionId, FuncRef>,
    tail_sig: SigRef,
}

impl<'a, 'f> RegionEmitter<'a, 'f> {
    fn emit(
        exec: &'a DecodedSegment,
        a: &'a Analysis,
        region: RegionId,
        func: &'f mut Function,
        fctx: &'f mut FunctionBuilderContext,
        isa: &OwnedTargetIsa,
    ) -> Option<()> {
        let r = a.regions.get(region)?;
        let mut used = [false; 32];
        let mut written = [false; 32];
        for &i in &r.insns {
            let (reads, write) = operands(a.insns.get(i)?.op);
            for s in reads.into_iter().flatten() {
                *used.get_mut(s as usize)? = true;
            }
            if let Some(d) = write {
                *used.get_mut(d as usize)? = true;
                *written.get_mut(d as usize)? = true;
            }
        }

        let mut b = FunctionBuilder::new(func, fctx);
        let regs: [Variable; 32] = std::array::from_fn(|_| b.declare_var(I64));
        let counter = b.declare_var(I64);
        let tail_sig = b.import_signature(tail_signature());

        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let (state, pc_in, counter_in) = match b.block_params(entry) {
            &[s, p, c] => (s, p, c),
            _ => return None,
        };
        let ld =
            |b: &mut FunctionBuilder<'_>, off: i32| b.ins().load(I64, host_flags(), state, off);
        let regs_ptr = ld(&mut b, JitState::OFFSET_REGS);
        let arena = ld(&mut b, JitState::OFFSET_ARENA);
        let read_base = ld(&mut b, JitState::OFFSET_READ_BASE);
        let read_len = ld(&mut b, JitState::OFFSET_READ_LEN);
        let write_base = ld(&mut b, JitState::OFFSET_WRITE_BASE);
        let write_len = ld(&mut b, JitState::OFFSET_WRITE_LEN);
        let max = ld(&mut b, JitState::OFFSET_MAX);
        for (n, (&u, &var)) in used.iter().zip(&regs).enumerate() {
            if u {
                let v = b.ins().load(I64, host_flags(), regs_ptr, (n * 8) as i32);
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
            b,
            regs,
            counter,
            written,
            state,
            regs_ptr,
            arena,
            read_base,
            read_len,
            write_base,
            write_len,
            max,
            labels: HashMap::new(),
            exit,
            exit_stored,
            xfer: HashMap::new(),
            indirect: None,
            funcs: HashMap::new(),
            tail_sig,
        };

        // Entry dispatch: one block per entry point adds the block's count.
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
            let c =
                e.b.ins()
                    .iadd_imm_s(counter_in, block_icount(exec, pc) as i64);
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

    fn reg(&mut self, r: u32) -> Value {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        let var = self.regs[(r & 31) as usize];
        self.b.use_var(var)
    }

    fn set_reg(&mut self, r: u32, v: Value) {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        let var = self.regs[(r & 31) as usize];
        self.b.def_var(var, v);
    }

    fn count(&mut self) -> Value {
        self.b.use_var(self.counter)
    }

    fn body(&mut self, insns: &[usize]) -> Option<()> {
        // `open`: the current block can still fall through.
        let mut open = false;
        for &i in insns {
            let ins = *self.insn(i)?;
            if let Some(&label) = self.labels.get(&ins.pc) {
                if open {
                    // Sequential flow into an entry point: no new block.
                    self.b.ins().jump(label, &[]);
                }
                self.b.switch_to_block(label);
            } else if !open {
                // Unreachable by construction: a non-entry instruction is
                // only ever reached by falling through.
                return None;
            }
            open = self.instruction(&ins)?;
            if open {
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
                    let c = self
                        .b
                        .ins()
                        .iadd_imm_s(c, block_icount(self.exec, target) as i64);
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
        let c = self
            .b
            .ins()
            .iadd_imm_s(c, (block_icount(self.exec, target) as i64).wrapping_neg());
        match self.a.region_at(self.exec, target) {
            Some(r) => self.tail_call(r, target, c),
            None => self.leave(EXIT_JUMP, target, c),
        }
    }

    /// `if counter >= max { halt at target }`, as the dispatcher checks
    /// before jumping.
    fn check_counter(&mut self, target: u64) {
        let c = self.count();
        let over = self
            .b
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, c, self.max);
        let ok = self.b.create_block();
        let kind = self.b.ins().iconst(I64, EXIT_HALT as i64);
        let pc = self.b.ins().iconst(I64, target as i64);
        self.b
            .ins()
            .brif(over, self.exit, &args(&[kind, pc, c]), ok, &[]);
        self.b.switch_to_block(ok);
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
        }
        self.b.ins().return_(&[]);

        let mut xfer: Vec<_> = self.xfer.iter().map(|(&r, &b)| (r, b)).collect();
        xfer.sort_unstable();
        for (region, blk) in xfer {
            self.b.switch_to_block(blk);
            let p = self.b.block_params(blk).to_vec();
            self.store_regs();
            let func = self.func_ref(region);
            if let &[pc, counter] = &p[..] {
                self.b.ins().return_call(func, &[self.state, pc, counter]);
            }
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
            self.b
                .ins()
                .return_call_indirect(self.tail_sig, func, &[st, target, counter]);

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

    /// Bounds-check `addr` against `[base, base + len)` and return the host
    /// address. A failed check is a protection fault, like `Memory`.
    fn guest_addr(&mut self, addr: Value, write: bool) -> Value {
        let (base, len) = if write {
            (self.write_base, self.write_len)
        } else {
            (self.read_base, self.read_len)
        };
        let off = self.b.ins().isub(addr, base);
        let ok = self.b.ins().icmp(IntCC::UnsignedLessThan, off, len);
        let cont = self.b.create_block();
        let kind = self.b.ins().iconst(I64, EXIT_FAULT as i64);
        let c = self.count();
        self.b
            .ins()
            .brif(ok, cont, &[], self.exit, &args(&[kind, addr, c]));
        self.b.switch_to_block(cont);
        self.b.ins().iadd(self.arena, addr)
    }

    /// Emit one instruction. Returns whether control falls through.
    fn instruction(&mut self, ins: &Insn) -> Option<bool> {
        let next = ins.pc.wrapping_add(ins.len);
        match ins.op {
            JOp::Nop => {}
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
                let base = self.reg(rs1);
                let addr = self.b.ins().iadd_imm_s(base, imm);
                let host = self.guest_addr(addr, false);
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
                let base = self.reg(rs1);
                let value = self.reg(rs2);
                let addr = self.b.ins().iadd_imm_s(base, imm);
                let host = self.guest_addr(addr, true);
                self.store(width, value, host);
            }
            JOp::Branch {
                cond,
                rs1,
                rs2,
                target,
                check,
            } => {
                let a = self.reg(rs1);
                let taken = match rs2 {
                    Some(rs2) => {
                        let b = self.reg(rs2);
                        self.b.ins().icmp(intcc(cond), a, b)
                    }
                    None => self.b.ins().icmp_imm_s(intcc(cond), a, 0),
                };
                let yes = self.b.create_block();
                let no = self.b.create_block();
                self.b.ins().brif(taken, yes, &[], no, &[]);
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
                self.enter_block(target);
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
                let c = self.count();
                if !COMPRESSED {
                    let low = self.b.ins().band_imm_s(target, 3);
                    let ok = self.b.create_block();
                    let kind = self.b.ins().iconst(I64, EXIT_MISALIGNED as i64);
                    self.b
                        .ins()
                        .brif(low, self.exit, &args(&[kind, target, c]), ok, &[]);
                    self.b.switch_to_block(ok);
                }
                // overflow_checked_jump
                let over = self
                    .b
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, c, self.max);
                let go = self.b.create_block();
                let kind = self.b.ins().iconst(I64, EXIT_HALT as i64);
                self.b
                    .ins()
                    .brif(over, self.exit, &args(&[kind, target, c]), go, &[]);
                self.b.switch_to_block(go);
                let blk = self.indirect_block();
                self.b.ins().jump(blk, &args(&[target, c]));
                return Some(false);
            }
        }
        Some(true)
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

/// The registers an instruction reads and the one it writes.
fn operands(op: JOp) -> ([Option<u32>; 2], Option<u32>) {
    match op {
        JOp::Nop | JOp::Fence => ([None, None], None),
        JOp::Const { rd, .. } => ([None, None], Some(rd)),
        JOp::AluImm { rd, rs1, .. } | JOp::Unary { rd, rs1, .. } => ([Some(rs1), None], Some(rd)),
        JOp::Alu { rd, rs1, rs2, .. } => ([Some(rs1), Some(rs2)], Some(rd)),
        JOp::Load { rd, rs1, .. } => ([Some(rs1), None], rd),
        JOp::Store { rs1, rs2, .. } => ([Some(rs1), Some(rs2)], None),
        JOp::Branch { rs1, rs2, .. } => ([Some(rs1), rs2], None),
        JOp::Jal { rd, .. } => ([None, None], (rd != 0).then_some(rd)),
        JOp::Jalr { rd, rs1, .. } => ([Some(rs1), None], (rd != 0).then_some(rd)),
    }
}
