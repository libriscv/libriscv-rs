//! Port of `asmjit/aj_api.cpp` and the `aj_translate.cpp` driver using Cranelift.
//! Requires `jit`; enable compilation with
//! [`MachineOptions::jit`](crate::MachineOptions::jit).
//!
//! New execute segments are analyzed into `codegen::region` regions and compiled
//! by [`emit`]. Eligible decoder block starts use [`native_enter`]. Native code
//! returns on unsupported instructions, faults, or instruction limits, preserving
//! interpreter PC, registers, and counts. [`cache`] saves output for embedding
//! or loading in later runs.
//!
//! This module enforces safe Rust. `libriscv-jit-rt` maps and enters native code.
//!
//! [`native_enter`]: crate::dispatch::native_enter

pub(crate) mod cache;
mod emit;

use libriscv_jit_rt::{CodeMemory, Host, HostCall, HostParts, MemoryLayout};
use std::sync::Arc;

pub(crate) use emit::Compiled;
use emit::Image;

use crate::codegen::region;
use crate::decoder_cache::DecodedSegment;
use crate::dispatch::{Counter, call_dyncall, dyncall_done};
use crate::error::{ExceptionKind, MachineError};
use crate::machine::Machine;
use crate::native_code::{EXIT_HOST, NativeCode, NativeExit, NativeSegment};

/// Compile `segment` for `layout` on a worker and publish the result through
/// [`DecodedSegment::finish_background_jit`]. Machines sharing the segment
/// continue interpreting until they install the result at a slice boundary.
/// If thread creation fails, compile synchronously. See `MachineOptions::jit_background`.
pub(crate) fn spawn_background(
    segment: &Arc<DecodedSegment>,
    layout: MemoryLayout,
    save: Option<Save>,
) {
    /// Publish an empty result on panic to release compilation waiters.
    /// Has no effect after a result is published.
    struct Finish(Arc<DecodedSegment>);
    impl Drop for Finish {
        fn drop(&mut self) {
            self.0.finish_background_jit(None);
        }
    }
    segment.begin_background_jit();
    // Writing a cache file: `cache::wait_for_saves` waits for it.
    let pending = save
        .as_ref()
        .filter(|s| s.file.is_some())
        .map(|_| cache::PendingSave::new());
    let finish = Finish(Arc::clone(segment));
    let fallback = save.clone();
    let spawned = std::thread::Builder::new()
        .name("libriscv-jit".into())
        .spawn(move || {
            let compiled = compile_and_load(&finish.0, layout, save.as_ref());
            finish.0.finish_background_jit(compiled);
            drop(pending);
        });
    if spawned.is_err() {
        let compiled = compile_and_load(segment, layout, fallback.as_ref());
        segment.finish_background_jit(compiled);
    }
}

/// Where to save the code compiled for a segment: to
/// `MachineOptions::jit_cache_output` and/or to a file of
/// `MachineOptions::jit_cache_dir`.
#[derive(Debug, Clone)]
pub(crate) struct Save {
    pub key: cache::CacheKey,
    pub output: Option<cache::JitCacheOutput>,
    pub file: Option<std::path::PathBuf>,
    pub verbose: bool,
}

/// How to compile a segment whose compilation was deferred until it is hot
/// ([`DecodedSegment::defer_jit`]).
#[derive(Debug)]
pub(crate) struct Deferred {
    pub layout: MemoryLayout,
    pub save: Option<Save>,
    /// On a new thread (`MachineOptions::jit_background`), or right away.
    pub background: bool,
}

/// Compile `segment`, whose compilation was deferred, publishing the result
/// as a background compilation does: every machine holding the segment
/// switches to the compiled copy at its next slice.
pub(crate) fn compile_deferred(segment: &Arc<DecodedSegment>, deferred: &Deferred) {
    if deferred.background {
        spawn_background(segment, deferred.layout, deferred.save.clone());
    } else {
        segment.begin_background_jit();
        let compiled = compile_and_load(segment, deferred.layout, deferred.save.as_ref());
        segment.finish_background_jit(compiled);
    }
}

/// Compile `exec` and patch its decoder entries to enter the native code
/// (saving the code to `save`). Leaves `exec` untouched if nothing could
/// be compiled.
pub(crate) fn attach(exec: &mut DecodedSegment, layout: MemoryLayout, save: Option<&Save>) {
    if let Some(compiled) = compile_and_load(exec, layout, save) {
        install(exec, compiled);
    }
}

/// Patch `exec`'s decoder entries to enter `compiled`.
pub(crate) fn install(exec: &mut DecodedSegment, compiled: Compiled) {
    NativeSegment::install(
        exec,
        NativeCode::Jit(Arc::new(compiled.code)),
        compiled.patch,
    );
}

/// Give `exec`, a segment rebuilt in place of the stale segment `old`, the
/// JIT code of `old` if it would compile to the same: the same bytes at the
/// same address, decoded the same way, for the same memory layout. True if
/// it did; then there is nothing to compile.
///
/// FENCE.I marks every execute segment stale, and the next instruction
/// fetch rebuilds the segment from memory (as in C++). Code that has not
/// changed would compile to the same thing again, which takes the JIT
/// milliseconds per page: a loop around a FENCE.I would cost that every
/// iteration, so that a guest's instruction limit no longer bounds the host
/// time it takes (found by `vmfuzzer`, 754 compiles in 1500 instructions).
pub(crate) fn reuse(exec: &mut DecodedSegment, old: &DecodedSegment, layout: MemoryLayout) -> bool {
    let Some((code, patch)) = old.native.as_ref().and_then(NativeSegment::jit_code) else {
        return false;
    };
    let same = old.exec_begin() == exec.exec_begin()
        && old.is_execute_only() == exec.is_execute_only()
        && old.cache.len() == exec.cache.len()
        && code.layout() == layout
        && old.code() == exec.code();
    if same {
        NativeSegment::install(exec, NativeCode::Jit(code), patch);
    }
    same
}

/// [`compile`], save the code to `save`, then map it.
fn compile_and_load(
    exec: &DecodedSegment,
    layout: MemoryLayout,
    save: Option<&Save>,
) -> Option<Compiled> {
    let image = compile(exec, layout)?;
    if let Some(save) = save {
        // Saved with the code it was compiled from, which cached code must
        // match to be used.
        if let Some(output) = &save.output {
            output.record(save.key, &image, exec.code());
        }
        if let Some(file) = &save.file {
            cache::save_file(save.key, &image, exec.code(), file, save.verbose);
        }
    }
    image.load()
}

/// Compile `exec` without changing it. Only reads the segment, so it can
/// run on another thread while the interpreter executes the segment.
/// With a non-zero `layout.mask` (the arena size minus one, a power of
/// two minus one), guest addresses are masked instead of bounds-checked
/// (`MachineOptions::jit_automatic_nbit_address_space`). Otherwise the
/// code checks them against `layout`'s windows, which are constants in
/// it (C++ `aj_machine_info`'s constant arena bounds); machines whose
/// windows do not contain them interpret the segment instead.
fn compile(exec: &DecodedSegment, layout: MemoryLayout) -> Option<Image> {
    #[cfg(test)]
    tests::COMPILES.with(|c| c.set(c.get() + 1));
    let analysis = region::analyse(exec);
    emit::compile(exec, &analysis, layout)
}

/// Run the JIT code `code` of the current segment `exec` from the block
/// starting at `pc` (decoder entry `slot`), with the counter as it was
/// *before* beginning that block.
pub(crate) fn run(
    code: &CodeMemory,
    exec: &DecodedSegment,
    m: &mut Machine,
    slot: usize,
    pc: u64,
    counter: u64,
    max: u64,
) -> NativeExit {
    let counts = !code.layout().unlimited;
    let mut host = JitHost {
        m,
        outcome: None,
        exec,
        counts,
    };
    let Some(exit) = code.enter(&mut host, slot, pc, counter, max) else {
        return NativeExit::Unavailable;
    };
    #[cfg(test)]
    if exit.kind == crate::native_code::EXIT_SLOW {
        tests::SLOW_EXITS.with(|n| n.set(n.get() + 1));
    }
    if exit.kind != EXIT_HOST {
        return NativeExit::from_raw(exit.kind, exit.pc, exit.counter);
    }
    match host.outcome {
        Some(Ok((c, touched))) => NativeExit::Dyncalled {
            pc: exit.pc,
            done: dyncall_done(exec, exit.pc, c, touched, counts),
        },
        Some(Err(err)) => NativeExit::Fault(err),
        None => NativeExit::Fault(MachineError::trigger(
            ExceptionKind::UnknownException,
            exit.pc,
        )),
    }
}

/// The machine as JIT code sees it: its registers and arena, and the host
/// calls it makes, which are dyncalls in the middle of their block (C++
/// asmjit `emit_dyncall`; one that ends its block leaves native code
/// instead, with `EXIT_DYNCALL`). The code stores the registers the
/// dyncall promises the handler reads (`codegen::calls::dyncall_regs`) and
/// calls [`Host::call`] with the dyncall word, which runs it as the
/// dispatcher's dyncall exit does (`dispatch::run_dyncall`). The code goes
/// on after it unless the handler faulted, stopped the machine (or
/// penalized it to the limit) or made the segment stale: then it leaves
/// with `EXIT_HOST`, and the dispatcher continues from `outcome` exactly
/// as after the dyncall exit.
struct JitHost<'m, 'e> {
    m: &'m mut Machine,
    /// The last dyncall's outcome: [`call_dyncall`]'s.
    outcome: Option<crate::Result<(Counter, bool)>>,
    exec: &'e DecodedSegment,
    /// The code counts instructions (`MemoryLayout::unlimited` is false).
    counts: bool,
}

impl Host for JitHost<'_, '_> {
    #[inline]
    fn parts(&mut self) -> HostParts<'_> {
        let MemoryLayout { read, write, .. } = self.m.memory.native_layout();
        let frm = u64::from(self.m.cpu.regs.fcsr().frm());
        let (regs, fregs) = self.m.cpu.regs.x_f_mut();
        let (arena, pages) = self.m.memory.native_memory();
        HostParts {
            regs,
            fregs,
            arena,
            read,
            write,
            frm,
            pages,
        }
    }

    #[inline]
    fn fill_page(&mut self, addr: u64, size: u64, store: bool) -> bool {
        self.m.memory.native_fill(addr, size, store)
    }

    #[inline]
    fn call(&mut self, call: &mut HostCall) -> bool {
        let c = Counter {
            value: call.counter,
            max: call.max,
        };
        call.exit_kind = EXIT_HOST;
        call.exit_pc = call.pc;
        let (c, touched) = match call_dyncall(self.m, call.pc, call.arg as u32, c) {
            Ok(done) => {
                self.outcome = Some(Ok(done));
                done
            }
            Err(err) => {
                self.outcome = Some(Err(err));
                return false;
            }
        };
        call.counter = c.value;
        call.max = c.max;
        if self.exec.is_stale() {
            return false;
        }
        // As in the `dyncall` bytecode: the retired count decides.
        !touched || {
            let d = dyncall_done(self.exec, call.pc, c, touched, self.counts);
            !d.halt_after.overflowed()
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Slicing while a background compilation is pending must not be
    //! observable: the same results, registers and counts as one
    //! uninterrupted run, and system calls see the real instruction limit.

    use crate::registers::*;
    use crate::{Machine, MachineOptions};

    const V: u64 = 0x10000;

    thread_local! {
        /// The PC of every entry into native code on this thread.
        pub(crate) static ENTRIES: std::cell::RefCell<Vec<u64>> =
            const { std::cell::RefCell::new(Vec::new()) };
        /// Native exits for a load or store the interpreter makes, on this
        /// thread.
        pub(crate) static SLOW_EXITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// Segments compiled on this thread.
        pub(crate) static COMPILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13
    }
    const fn bne(rs1: u32, rs2: u32, off: i32) -> u32 {
        let o = off as u32;
        ((o >> 12 & 1) << 31)
            | ((o >> 5 & 0x3F) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (1 << 12)
            | ((o >> 1 & 0xF) << 8)
            | ((o >> 11 & 1) << 7)
            | 0x63
    }
    const fn ld(rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (3 << 12) | (rd << 7) | 0x03
    }
    const fn sd(rs1: u32, rs2: u32, imm: i32) -> u32 {
        let i = imm as u32;
        ((i >> 5 & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (3 << 12) | ((i & 0x1F) << 7) | 0x23
    }
    const ECALL: u32 = 0x73;
    const JSELF: u32 = 0x6F;

    /// a0 = the limit the handler sees, a1 accumulates it with the counter.
    fn observe(m: &mut Machine) -> crate::Result<()> {
        let (max, counter) = (m.max_instructions(), m.instruction_counter());
        m.cpu.set_reg(REG_ARG0, max);
        let acc = m.cpu.reg(REG_ARG1);
        m.cpu.set_reg(REG_ARG1, acc.rotate_left(7) ^ max ^ counter);
        Ok(())
    }
    fn stop(m: &mut Machine) -> crate::Result<()> {
        m.stop();
        Ok(())
    }
    fn extend(m: &mut Machine) -> crate::Result<()> {
        let c = m.instruction_counter();
        m.set_max_instructions(c + 77);
        Ok(())
    }

    /// A loop with system calls, then `final_syscall` (or with 0, a read
    /// fault), then a spin.
    fn program(final_syscall: i32) -> Vec<u8> {
        // lw s1, 0(x0)
        let last = if final_syscall == 0 {
            (9 << 7) | (2 << 12) | 0x03
        } else {
            ECALL
        };
        [
            addi(5, 0, 300),
            addi(REG_ECALL, 0, 500),
            // loop:
            addi(6, 6, 7),
            ECALL,
            addi(7, 7, 1),
            addi(5, 5, -1),
            bne(5, 0, -16),
            addi(REG_ECALL, 0, final_syscall),
            last,
            addi(8, 8, 1),
            JSELF,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
    }

    fn run(code: &[u8], slice: Option<u64>, fuel: u64) -> (Result<bool, String>, Vec<u64>) {
        let mut m = Machine::new(
            [].into(),
            MachineOptions {
                memory_max: 1 << 20,
                jit: slice.is_some(),
                jit_background: true,
                jit_background_slice: slice.unwrap_or(1),
                ..Default::default()
            },
        )
        .unwrap();
        m.install_syscall_handler(500, observe);
        m.install_syscall_handler(501, stop);
        m.install_syscall_handler(502, extend);
        m.init_execute_area(code, V).unwrap();
        if slice.is_some() {
            m.memory.hold_jit_pending();
            assert!(m.jit_pending());
        }
        m.cpu.jump(V).unwrap();
        let result = m.try_simulate(fuel, 0).map_err(|e| e.to_string());
        let mut state = m.cpu.registers().all().to_vec();
        state.extend([m.cpu.pc(), m.instruction_counter(), m.max_instructions()]);
        (result, state)
    }

    /// Runs `page` at `V + 0x1000` (read, write, execute; reached from a
    /// main segment at `V`, which never goes stale) with `t2` pointing at
    /// it: the state, and the compilations it took on this thread.
    fn run_page(
        page: &[u32],
        jit: bool,
        background: bool,
        slice: u64,
        fuel: u64,
    ) -> ((Result<bool, String>, Vec<u64>), usize) {
        let lp = V + 0x1000;
        let bytes: Vec<u8> = page.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut m = Machine::new(
            [].into(),
            MachineOptions {
                memory_max: 1 << 20,
                jit,
                jit_background: background,
                jit_background_slice: slice,
                // Counting compiles: build every segment here, not take
                // the entry stub (the same in other tests) from a machine
                // of a concurrent test.
                use_shared_execute_segments: false,
                ..Default::default()
            },
        )
        .unwrap();
        m.memory.memcpy(lp, &bytes).unwrap();
        let rwx = crate::memory::PageAttributes {
            read: true,
            write: true,
            exec: true,
            ..Default::default()
        };
        m.memory.set_page_attr(lp, 0x1000, rwx).unwrap();
        m.cpu.set_reg(7, lp);
        let before = COMPILES.with(std::cell::Cell::get);
        // lui t0, lp; jalr x0, 0(t0)
        let entry: Vec<u8> = [
            (lp as u32 & 0xFFFF_F000) | (5 << 7) | 0x37,
            (5 << 15) | 0x67,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        m.init_execute_area(&entry, V).unwrap();
        m.cpu.jump(V).unwrap();
        let result = m.try_simulate(fuel, 0).map_err(|e| e.to_string());
        if background {
            m.wait_for_jit();
        }
        let compiles = COMPILES.with(std::cell::Cell::get) - before;
        let mut state = m.cpu.registers().all().to_vec();
        state.extend([m.cpu.pc(), m.instruction_counter()]);
        if let Ok(page) = m.memory.memview(lp, 0x100) {
            state.extend(page.iter().map(|&b| u64::from(b)));
        }
        ((result, state), compiles)
    }

    const FENCE_I: u32 = 0x100F;
    /// `sw t1, 0x40(t2)`: changes the page's code bytes (past the code).
    const SW_T1_PAGE: u32 = (2 << 25) | (6 << 20) | (7 << 15) | (2 << 12) | 0x23;

    /// Code that changes its own page and runs FENCE.I in a loop: every
    /// iteration rebuilds the page from changed bytes. The rebuilds are
    /// interpreted (each would take a compilation of milliseconds) until
    /// one stays valid for `jit_background_slice` instructions, which none
    /// does here. (Nor do the instructions charged for the rebuilds make
    /// them hot.)
    #[test]
    fn code_rewritten_in_a_loop_is_not_compiled_each_time() {
        const J_BACK_12: u32 = 0xFF5F_F06F; // jal x0, -12
        // Each rebuild is charged 1024 instructions (a page).
        const FUEL: u64 = 2_000_000;
        let page = [SW_T1_PAGE, addi(6, 6, 1), FENCE_I, J_BACK_12];
        let (expected, _) = run_page(&page, false, false, 1 << 20, FUEL);
        assert!(expected.1[6] > 1500, "{expected:?}");
        for background in [false, true] {
            let (got, compiles) = run_page(&page, true, background, 1 << 20, FUEL);
            assert_eq!(got, expected, "background {background}");
            if super::emit::HOST_SUPPORTED && !background {
                // The main segment and the page's first segment.
                assert_eq!(compiles, 2);
            }
        }
    }

    /// A rebuild from changed code that stays valid is compiled once it is
    /// hot, and runs natively from then on.
    #[test]
    fn a_rebuilt_segment_that_stays_valid_is_compiled_once_hot() {
        const J_BACK_4: u32 = 0xFFDF_F06F; // jal x0, -4
        // Change the page once (t1 is not 0, so its bytes do change), then
        // loop.
        let page = [
            addi(6, 0, 0x55),
            SW_T1_PAGE,
            FENCE_I,
            addi(10, 10, 1),
            J_BACK_4,
        ];
        let (expected, _) = run_page(&page, false, false, 1000, 20_000);
        for background in [false, true] {
            ENTRIES.with_borrow_mut(Vec::clear);
            let (got, compiles) = run_page(&page, true, background, 1000, 20_000);
            assert_eq!(got, expected, "background {background}");
            // In the background the code may arrive after the run: only the
            // results are compared then.
            if super::emit::HOST_SUPPORTED && !background {
                // The main segment, the page, and its rebuild, which then
                // runs natively.
                assert_eq!(compiles, 3);
                let loop_pc = V + 0x1000 + 12;
                assert!(ENTRIES.with_borrow(|e| e.contains(&loop_pc)));
            }
        }
    }

    /// A `jalr` after a constant whose target is the `jalr` itself: the
    /// second time, it is entered with the link it wrote as its base, and
    /// must jump with that (found by `jitdifffuzzer`).
    #[test]
    fn a_constant_jalr_reached_again_uses_its_base_register() {
        const LUI_A4: u32 = (V as u32) | (14 << 7) | 0x37; // lui a4, V
        const JALR_A4_4: u32 = (4 << 20) | (14 << 15) | (14 << 7) | 0x67; // jalr a4, 4(a4)
        // V+4 jumps to itself, then to V+12.
        let code: Vec<u8> = [LUI_A4, JALR_A4_4, JSELF, addi(10, 10, 1), JSELF]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let run = |jit: bool| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit,
                    jit_background: false,
                    ..Default::default()
                },
            )
            .unwrap();
            m.init_execute_area(&code, V).unwrap();
            m.cpu.jump(V).unwrap();
            let result = m.try_simulate(100, 0).map_err(|e| e.to_string());
            (
                result,
                m.cpu.reg(10),
                m.cpu.reg(14),
                m.cpu.pc(),
                m.instruction_counter(),
            )
        };
        let interpreted = run(false);
        assert_eq!(interpreted.1, 1, "{interpreted:?}");
        assert_eq!(run(true), interpreted);
    }

    /// A loop around a FENCE.I rebuilds its segment every iteration (one
    /// made from executable pages: the main segment is never stale). The
    /// rebuilds reuse the JIT code instead of compiling the unchanged page
    /// again (each compile takes milliseconds), with the results of the
    /// interpreter. (Decoding is free here, so that the loop iterates
    /// many times within a small instruction limit.)
    #[test]
    fn fence_i_loops_reuse_the_compiled_code() {
        const FENCE_I: u32 = 0x100F;
        const J_BACK_8: u32 = 0xFF9F_F06F; // jal x0, -8
        let lp = V + 0x1000;
        let mut page = vec![0u8; 0x1000];
        for (i, w) in [addi(6, 6, 1), FENCE_I, J_BACK_8].iter().enumerate() {
            page[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        let run = |jit: bool| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit,
                    jit_background: false,
                    free_decode_bytes: u64::MAX,
                    // Counting compiles: as in `run_page`.
                    use_shared_execute_segments: false,
                    ..Default::default()
                },
            )
            .unwrap();
            m.memory.memcpy(lp, &page).unwrap();
            let rx = crate::memory::PageAttributes {
                read: true,
                write: false,
                exec: true,
                ..Default::default()
            };
            m.memory.set_page_attr(lp, 0x1000, rx).unwrap();
            let before = COMPILES.with(std::cell::Cell::get);
            // lui t0, lp; jalr x0, 0(t0)
            let entry: Vec<u8> = [
                (lp as u32 & 0xFFFF_F000) | (5 << 7) | 0x37,
                (5 << 15) | 0x67,
            ]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
            m.init_execute_area(&entry, V).unwrap();
            m.cpu.jump(V).unwrap();
            let result = m.try_simulate(3001, 0).map_err(|e| e.to_string());
            let compiles = COMPILES.with(std::cell::Cell::get) - before;
            let state = (result, m.cpu.reg(6), m.cpu.pc(), m.instruction_counter());
            (state, compiles)
        };
        let (interpreted, _) = run(false);
        let (compiled, compiles) = run(true);
        assert_eq!(compiled, interpreted);
        assert!(interpreted.1 > 500, "{interpreted:?}");
        if super::emit::HOST_SUPPORTED {
            // The main segment and the loop's page, once.
            assert_eq!(compiles, 2);
        }
    }

    /// F and D are compiled, apart from the formats and Zfa/Zfhmin
    /// encodings the interpreter rejects or handles in software.
    #[test]
    #[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
    fn fp_instructions_are_compiled() {
        const fn fp(fpfunc: u32, fmt: u32, rm: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
            (((fpfunc << 2) | fmt) << 25)
                | (rs2 << 20)
                | (rs1 << 15)
                | (rm << 12)
                | (rd << 7)
                | 0x53
        }
        let native = [
            (2 << 12) | (1 << 7) | (10 << 15) | 0x07,  // flw ft1, 0(a0)
            (3 << 12) | (1 << 7) | (10 << 15) | 0x07,  // fld ft1, 0(a0)
            (3 << 12) | (1 << 20) | (10 << 15) | 0x27, // fsd ft1, 0(a0)
            (3 << 27) | (1 << 25) | (2 << 20) | (1 << 15) | 0x43, // fmadd.d
            (3 << 27) | (2 << 20) | (1 << 15) | 0x4F,  // fnmadd.s
            fp(0x00, 1, 7, 1, 2, 3),                   // fadd.d
            fp(0x03, 0, 0, 1, 2, 3),                   // fdiv.s
            fp(0x04, 1, 2, 1, 2, 3),                   // fsgnjx.d
            fp(0x05, 0, 1, 1, 2, 3),                   // fmax.s
            fp(0x0B, 1, 0, 1, 2, 0),                   // fsqrt.d
            fp(0x08, 0, 0, 1, 2, 1),                   // fcvt.s.d
            fp(0x08, 1, 0, 1, 2, 0),                   // fcvt.d.s
            fp(0x14, 1, 2, 10, 2, 3),                  // feq.d a0
            fp(0x14, 1, 2, 0, 2, 3),                   // feq.d x0 (a NOP)
            fp(0x18, 1, 4, 10, 2, 0),                  // fcvt.w.d a0, rmm
            fp(0x18, 0, 7, 10, 2, 3),                  // fcvt.lu.s a0, dyn
            fp(0x1A, 1, 0, 1, 10, 3),                  // fcvt.d.lu
            fp(0x1C, 1, 0, 10, 2, 0),                  // fmv.x.d
            fp(0x1C, 0, 1, 10, 2, 0),                  // fclass.s
            fp(0x1E, 0, 0, 1, 10, 0),                  // fmv.w.x
        ];
        let interpreted = [
            (1 << 12) | (1 << 7) | (10 << 15) | 0x07,     // flh (Zfhmin)
            fp(0x00, 2, 0, 1, 2, 3),                      // fadd.h
            fp(0x05, 1, 2, 1, 2, 3),                      // fminm.d (Zfa)
            fp(0x08, 0, 0, 1, 2, 4),                      // fround.s (Zfa)
            fp(0x08, 0, 0, 1, 2, 2),                      // fcvt.s.h
            fp(0x18, 1, 1, 10, 2, 8),                     // fcvtmod.w.d (Zfa)
            fp(0x1E, 1, 0, 1, 1, 1),                      // fli.d (Zfa)
            (0x003 << 20) | (2 << 12) | (10 << 7) | 0x73, // frcsr a0
        ];
        for (words, expect) in [(&native[..], true), (&interpreted[..], false)] {
            for &w in words {
                let code: Vec<u8> = [w, JSELF].iter().flat_map(|w| w.to_le_bytes()).collect();
                let mut m = Machine::new(
                    [].into(),
                    MachineOptions {
                        memory_max: 1 << 20,
                        ..Default::default()
                    },
                )
                .unwrap();
                m.init_execute_area(&code, V).unwrap();
                let a = crate::codegen::region::analyse(m.cpu.current_execute_segment());
                let compiled = a.insns.iter().any(|i| i.pc == V);
                assert_eq!(compiled, expect, "{w:#010x}");
            }
        }
    }

    /// Without SSE4.1 and FMA, rounding and FMA are calls to the helpers in
    /// `emit`: the same results as the interpreter. (AArch64 has both.)
    #[test]
    #[cfg(all(
        feature = "ext-f",
        not(feature = "debug-checks"),
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    fn fp_libcalls_match_the_interpreter() {
        use std::sync::atomic::Ordering;
        const fn fp(fpfunc: u32, fmt: u32, rm: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
            (((fpfunc << 2) | fmt) << 25)
                | (rs2 << 20)
                | (rs1 << 15)
                | (rm << 12)
                | (rd << 7)
                | 0x53
        }
        let mut code = Vec::new();
        for rm in 0..8 {
            code.push(fp(0x18, 1, rm, 10 + rm, 1, 2)); // fcvt.l.d a(rm), ft1, rm
            code.push(fp(0x18, 0, rm, 18 + rm % 6, 2, 0)); // fcvt.w.s s(..), ft2, rm
        }
        code.push((3 << 27) | (1 << 25) | (2 << 20) | (1 << 15) | (4 << 7) | 0x43); // fmadd.d ft4
        code.push((3 << 27) | (2 << 20) | (1 << 15) | (5 << 7) | 0x4B); // fnmsub.s ft5
        code.push(JSELF);
        let code: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let run = |jit: bool, a: f64, b: f32| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit,
                    ..Default::default()
                },
            )
            .unwrap();
            m.init_execute_area(&code, V).unwrap();
            if jit {
                assert!(m.cpu.current_execute_segment().jit_entry_count() > 0);
            }
            m.cpu.registers_mut().set_fp_bits(1, a.to_bits());
            m.cpu.registers_mut().set_fp_bits(2, u64::from(b.to_bits()));
            m.cpu.registers_mut().set_fp_bits(3, 0.1f64.to_bits());
            m.cpu.registers_mut().fcsr_mut().set_frm(3);
            m.cpu.jump(V).unwrap();
            let _ = m.try_simulate(100, 0);
            (
                m.cpu.registers().all().to_vec(),
                m.cpu.registers().all_fp().to_vec(),
            )
        };
        let _baseline = BASELINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        super::emit::BASELINE_ISA.store(true, Ordering::Relaxed);
        for (a, b) in [
            (2.5, -2.5f32),
            (-3.5, 3.5),
            (1e15 + 0.5, 0.49999997),
            (-0.5, -1.5),
        ] {
            assert_eq!(run(true, a, b), run(false, a, b), "{a} {b}");
        }
        super::emit::BASELINE_ISA.store(false, Ordering::Relaxed);
    }

    /// Serializes the tests that compile for baseline x86-64.
    #[cfg(target_arch = "x86_64")]
    static BASELINE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Saved code with libcalls (baseline x86-64: rounding and FMA are
    /// calls) links them against the C library's `nearbyint`, `fma` and so
    /// on, through the GOT: the same results as the interpreter.
    #[test]
    #[cfg(all(
        feature = "ext-f",
        not(feature = "debug-checks"),
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    fn saved_libcalls_match_the_interpreter() {
        use std::sync::atomic::Ordering;
        if !std::process::Command::new("cc")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            assert!(
                std::env::var("LIBRISCV_REQUIRE_TOOLCHAIN").is_err(),
                "cc is required"
            );
            return;
        }
        const fn fp(fpfunc: u32, fmt: u32, rm: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
            (((fpfunc << 2) | fmt) << 25)
                | (rs2 << 20)
                | (rs1 << 15)
                | (rm << 12)
                | (rd << 7)
                | 0x53
        }
        // Its own address and code: the registry is process-wide.
        const AT: u64 = 0x30000;
        let mut code = vec![addi(7, 7, 1)];
        for rm in 0..5 {
            code.push(fp(0x18, 1, rm, 10 + rm, 1, 2)); // fcvt.l.d a(rm), ft1, rm
        }
        code.push((3 << 27) | (1 << 25) | (2 << 20) | (1 << 15) | (4 << 7) | 0x43); // fmadd.d ft4
        code.push(JSELF);
        let code: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let run = |options: MachineOptions| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    ..options
                },
            )
            .unwrap();
            m.init_execute_area(&code, AT).unwrap();
            m.cpu.registers_mut().set_fp_bits(1, (-2.5f64).to_bits());
            m.cpu.registers_mut().set_fp_bits(2, 1e15f64.to_bits());
            m.cpu.registers_mut().set_fp_bits(3, 0.1f64.to_bits());
            m.cpu.jump(AT).unwrap();
            let _ = m.try_simulate(100, 0);
            let cached = m.cpu.current_execute_segment().jit_is_cached();
            let mut s = m.cpu.registers().all().to_vec();
            s.extend(m.cpu.registers().all_fp());
            (s, cached)
        };
        let output = super::cache::JitCacheOutput::new();
        {
            let _baseline = BASELINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            super::emit::BASELINE_ISA.store(true, Ordering::Relaxed);
            run(MachineOptions {
                jit: true,
                jit_cache_output: Some(output.clone()),
                ..Default::default()
            });
            super::emit::BASELINE_ISA.store(false, Ordering::Relaxed);
        }
        assert_eq!(output.len(), 1);
        let object = output.object().unwrap();
        for name in ["nearbyint", "ceil", "fma"] {
            let name = format!("\0{name}\0");
            assert!(
                object.windows(name.len()).any(|w| w == name.as_bytes()),
                "{name}"
            );
        }
        let dir = std::env::temp_dir().join(format!("libriscv-jit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let so = dir.join("libcalls.so");
        output.write_shared_object(&so).unwrap();
        assert_eq!(super::cache::load_shared_object(&so).unwrap(), 1);
        let _ = std::fs::remove_dir_all(&dir);
        let (expected, _) = run(MachineOptions {
            jit_cache: false,
            ..Default::default()
        });
        let (got, cached) = run(MachineOptions::default());
        assert!(cached);
        assert_eq!(got, expected);
    }

    /// Large segments are compiled on several threads; the image must be
    /// the one a single thread compiles.
    /// Guest accesses and entry table lookups take their address through a
    /// conditional move (`select_spectre_guard`), so that a mispredicted
    /// bounds check cannot reach outside the arena or the table, even
    /// speculatively.
    #[test]
    fn accesses_and_table_lookups_are_guarded_against_speculation() {
        let code: Vec<u8> = [
            (2 << 12) | (11 << 7) | (10 << 15) | 0x03,  // lw a1, 0(a0)
            (2 << 12) | (11 << 20) | (13 << 15) | 0x23, // sw a1, 0(a3)
            (12 << 15) | 0x67,                          // jalr x0, 0(a2)
            JSELF,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        // The `select_spectre_guard`s in the code compiled with or without
        // `jit_spectre_guards`, or `None` without a JIT.
        let guards = |on: bool| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit_spectre_guards: on,
                    ..Default::default()
                },
            )
            .unwrap();
            m.init_execute_area(&code, V).unwrap();
            let exec = m.cpu.current_execute_segment();
            let a = crate::codegen::region::analyse(exec);
            super::emit::CLIF.with_borrow_mut(|c| *c = Some(Vec::new()));
            let image = super::emit::compile_on(exec, &a, m.memory.native_layout(), 1);
            let clif = super::emit::CLIF
                .with_borrow_mut(Option::take)
                .unwrap_or_default();
            image.map(|_| clif.concat().matches("select_spectre_guard").count())
        };
        let Some(on) = guards(true) else {
            return; // no JIT on this host
        };
        // The load's check, the store's (another base register) and the
        // indirect jump's table lookup.
        assert!(on >= 3, "{on} guards");
        assert_eq!(guards(false), Some(0));
    }

    #[test]
    fn compiling_on_several_threads_gives_the_same_image() {
        // Many regions: compiled ADDIs separated by system calls (which
        // are not compiled), with forward branches between them.
        let mut words = Vec::new();
        for k in 0..400u32 {
            words.push(0x0015_0513 | ((k % 7) << 20)); // addi a0, a0, 1 + k % 7
            words.push(0x0000_0463); // beqz zero, +8
            words.push(0x0000_0013); // nop
            words.push(0x0000_0073); // ecall
        }
        words.push(JSELF);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut m = Machine::new(
            [].into(),
            MachineOptions {
                memory_max: 1 << 20,
                ..Default::default()
            },
        )
        .unwrap();
        m.init_execute_area(&code, V).unwrap();
        let exec = m.cpu.current_execute_segment();
        let a = crate::codegen::region::analyse(exec);
        assert!(a.regions.len() >= 400, "{} regions", a.regions.len());
        let layout = m.memory.native_layout();
        let one = super::emit::compile_on(exec, &a, layout, 1);
        assert_eq!(one.is_some(), super::emit::HOST_SUPPORTED);
        for threads in [2, 8, 64] {
            let many = super::emit::compile_on(exec, &a, layout, threads);
            assert_eq!(format!("{one:?}"), format!("{many:?}"), "{threads} threads");
        }
    }

    /// A loop head inside a decoder block (straight-line code runs into
    /// it) is a native entry, but its decoder entry is not patched, since the
    /// interpreter falls through it. A block beginning there must still
    /// enter native code: resuming after a halt at the back-edge, and after
    /// a background compilation arrives mid-loop (the slice ends at the
    /// back-edge). Otherwise the interpreter runs the rest of the loop.
    #[test]
    fn resuming_at_a_loop_head_inside_a_block_enters_native_code() {
        let head = V + 8;
        let code: Vec<u8> = [
            addi(15, 0, 1000),
            addi(10, 0, 0),
            // loop (inside the decoder block that began at V):
            addi(10, 10, 1),
            addi(15, 15, -1),
            bne(15, 0, -8),
            JSELF,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let state = |m: &Machine| {
            let mut s = m.cpu.registers().all().to_vec();
            s.extend([m.cpu.pc(), m.instruction_counter()]);
            s
        };
        let machine = |jit: bool, background: bool| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit,
                    jit_background: background,
                    jit_background_slice: 7,
                    ..Default::default()
                },
            )
            .unwrap();
            m.init_execute_area(&code, V).unwrap();
            m.cpu.jump(V).unwrap();
            m
        };
        let mut expected = machine(false, false);
        expected.try_simulate(100, 0).unwrap();
        assert_eq!(expected.cpu.pc(), head);
        let resume = |m: &mut Machine| {
            let c = m.instruction_counter();
            m.try_simulate(c + 2000, c).unwrap();
        };
        resume(&mut expected);

        for background in [false, true] {
            let mut m = machine(true, background);
            if background {
                // Interpret in slices until the halt, then let the code in.
                m.memory.hold_jit_pending();
            }
            m.try_simulate(100, 0).unwrap();
            assert_eq!(m.cpu.pc(), head, "background {background}");
            if background {
                m.memory.release_jit_pending();
                m.wait_for_jit();
            }
            ENTRIES.with_borrow_mut(Vec::clear);
            resume(&mut m);
            let first = ENTRIES.with_borrow(|e| e.first().copied());
            assert_eq!(first, Some(head), "background {background}");
            assert_eq!(state(&m), state(&expected), "background {background}");
        }
    }

    /// A CSR instruction ends its block and checks the limit after it ran;
    /// at the real limit the machine halts *at* it, as C++ does. A slice
    /// ending there must not do the same, or the next slice runs (and
    /// counts) the instruction again. Vectorised string functions read the
    /// `vl` CSR on every iteration.
    #[test]
    fn slices_dont_rerun_csr_instructions() {
        // csrrs rd, instret, x0
        const fn rdinstret(rd: u32) -> u32 {
            (0xC02 << 20) | (2 << 12) | (rd << 7) | 0x73
        }
        let code: Vec<u8> = [
            addi(5, 0, 300),
            // loop:
            rdinstret(11),
            addi(6, 6, 3),
            rdinstret(12),
            addi(5, 5, -1),
            bne(5, 0, -16),
            addi(8, 8, 1),
            JSELF,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let mut fuels: Vec<u64> = (0..30).collect();
        fuels.extend([100, 101, 102, 103, 1499, 1500, 1501, 1502, 1_000_000]);
        for fuel in fuels {
            let expected = run(&code, None, fuel);
            for slice in [1, 2, 3, 5, 16, 100] {
                let got = run(&code, Some(slice), fuel);
                assert_eq!(got, expected, "fuel {fuel}, slice {slice}");
            }
        }
    }

    #[test]
    fn slices_are_not_observable() {
        for final_syscall in [501, 502, 0] {
            let code = program(final_syscall);
            let mut fuels: Vec<u64> = (0..40).collect();
            fuels.extend([
                100, 999, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1600, 1700, 1_000_000,
            ]);
            for fuel in fuels {
                let expected = run(&code, None, fuel);
                for slice in [1, 2, 3, 5, 16, 100, 1 << 20] {
                    let got = run(&code, Some(slice), fuel);
                    assert_eq!(
                        got, expected,
                        "syscall {final_syscall}, fuel {fuel}, slice {slice}"
                    );
                }
            }
        }
    }

    /// Without an arena (forks and their masters), loads and stores go
    /// through the native page cache: a loop over two pages stays in native
    /// code, entered as often as with an arena (once per dispatch slice),
    /// with a page cache fill per page and kind instead of an interpreter
    /// exit per access.
    #[test]
    fn paged_loads_and_stores_stay_native() {
        let data = 0x40000u64;
        let code: Vec<u8> = [
            addi(15, 0, 1000),
            // loop: two pages, read and written
            ld(10, 5, 0),
            addi(10, 10, 1),
            sd(5, 10, 0),
            ld(11, 6, 8),
            addi(11, 11, 2),
            sd(6, 11, 8),
            addi(15, 15, -1),
            bne(15, 0, -28),
            JSELF,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let run = |jit: bool, arena: bool| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit,
                    use_memory_arena: arena,
                    ..Default::default()
                },
            )
            .unwrap();
            m.init_execute_area(&code, V).unwrap();
            m.cpu.set_reg(5, data);
            m.cpu.set_reg(6, data + 0x1000);
            m.cpu.jump(V).unwrap();
            ENTRIES.with_borrow_mut(Vec::clear);
            SLOW_EXITS.with(|n| n.set(0));
            let r = m.try_simulate(20_000, 0);
            let entries = (
                ENTRIES.with_borrow(Vec::len),
                SLOW_EXITS.with(std::cell::Cell::get),
            );
            let state = (
                r.map_err(|e| e.to_string()),
                *m.cpu.registers().all(),
                m.instruction_counter(),
                m.memory.read_u64(data).unwrap(),
                m.memory.read_u64(data + 0x1008).unwrap(),
            );
            (state, entries)
        };
        let (expected, _) = run(false, false);
        let (got, entries) = run(true, false);
        assert_eq!(got, expected);
        assert_eq!((got.3, got.4), (1000, 2000));
        let (with_arena, arena_entries) = run(true, true);
        assert_eq!(with_arena, expected);
        if crate::jit::emit::HOST_SUPPORTED {
            assert!(arena_entries.0 > 0);
            assert_eq!(arena_entries.1, 0);
            assert_eq!(entries, arena_entries);
        }
    }

    /// A store to a write-only page fills the page cache's store half only:
    /// a load from it then faults in JIT code as in the interpreter, rather
    /// than reading the bytes the store made reachable.
    #[test]
    fn paged_write_only_pages_stay_unreadable() {
        let data = 0x40000u64;
        let code: Vec<u8> = [addi(10, 0, 42), sd(5, 10, 0), ld(11, 5, 0), JSELF]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let run = |jit: bool| {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit,
                    use_memory_arena: false,
                    ..Default::default()
                },
            )
            .unwrap();
            m.init_execute_area(&code, V).unwrap();
            let write_only = crate::memory::PageAttributes {
                read: false,
                ..crate::memory::PageAttributes::default()
            };
            m.memory.set_page_attr(data, 0x1000, write_only).unwrap();
            m.cpu.set_reg(5, data);
            m.cpu.jump(V).unwrap();
            let r = m.try_simulate(1000, 0);
            (
                r.map_err(|e| (e.kind, e.to_string())),
                *m.cpu.registers().all(),
                m.instruction_counter(),
            )
        };
        let expected = run(false);
        assert!(expected.0.is_err(), "{expected:?}");
        assert_eq!(run(true), expected);
    }
}
