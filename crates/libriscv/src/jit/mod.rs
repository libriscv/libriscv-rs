//! Port of `asmjit/aj_api.cpp` and the driver of `aj_translate.cpp`, on
//! Cranelift instead of asmjit: the JIT. Built with the `jit` feature and
//! enabled per machine with [`MachineOptions::jit`](crate::MachineOptions::jit).
//!
//! When an execute segment is created, its compilable instructions are
//! grouped into regions (`codegen::region`) and compiled ([`emit`]).
//! Decoder entries at block starts that a region can be entered at get the
//! [`native_enter`] handler, so the interpreter enters native code the
//! first time it begins such a block, and native code runs until it hits something it doesn't
//! compile, a fault or the instruction limit. Then the interpreter resumes
//! with exactly the PC, registers and instruction count it would have had.
//!
//! The code compiled for a segment can be saved and used again by later
//! runs, linked into the program or loaded from a shared object; see
//! [`cache`].
//!
//! This module has no `unsafe` code: placing the code in executable memory
//! and entering it is done by `libriscv-jit-rt`.
//!
//! [`native_enter`]: crate::dispatch::native_enter

pub(crate) mod cache;
mod emit;

use libriscv_jit_rt::{CodeMemory, JitState, MemoryLayout};
use std::sync::Arc;

pub(crate) use emit::Compiled;
use emit::Image;

use crate::codegen::region;
use crate::decoder_cache::DecodedSegment;
use crate::machine::Machine;
use crate::native_code::{NativeCode, NativeExit, NativeSegment};

/// Compile `segment` (for the guest memory layout `layout`, see
/// [`compile`]) on a new thread (`MachineOptions::jit_background`,
/// C++ `asmjit_background_callback`), which publishes the result in the
/// segment ([`DecodedSegment::finish_background_jit`]). Every machine
/// holding the segment (forks, and machines sharing it through the
/// process-wide cache) keeps interpreting it meanwhile, and switches to the
/// compiled copy at its next slice. If no thread can be started, compiles
/// synchronously instead.
pub(crate) fn spawn_background(
    segment: &Arc<DecodedSegment>,
    layout: MemoryLayout,
    save: Option<Save>,
) {
    /// Publishes "nothing compiled" if the compilation panics, so that
    /// nobody waits for it forever. (No effect after a real result.)
    struct Finish(Arc<DecodedSegment>);
    impl Drop for Finish {
        fn drop(&mut self) {
            self.0.finish_background_jit(None);
        }
    }
    segment.begin_background_jit();
    let finish = Finish(Arc::clone(segment));
    let fallback = save.clone();
    let spawned = std::thread::Builder::new()
        .name("libriscv-jit".into())
        .spawn(move || {
            let compiled = compile_and_load(&finish.0, layout, save.as_ref());
            finish.0.finish_background_jit(compiled);
        });
    if spawned.is_err() {
        let compiled = compile_and_load(segment, layout, fallback.as_ref());
        segment.finish_background_jit(compiled);
    }
}

/// Where to save the code compiled for a segment (`MachineOptions::jit_cache_output`).
#[derive(Debug, Clone)]
pub(crate) struct Save {
    pub key: cache::CacheKey,
    pub output: cache::JitCacheOutput,
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
    NativeSegment::install(exec, NativeCode::Jit(compiled.code), compiled.patch);
}

/// [`compile`], save the code to `save`, then map it.
fn compile_and_load(
    exec: &DecodedSegment,
    layout: MemoryLayout,
    save: Option<&Save>,
) -> Option<Compiled> {
    let image = compile(exec, layout)?;
    if let Some(save) = save {
        save.output.record(save.key, &image);
    }
    image.load()
}

/// Compile `exec` without changing it. Only reads the segment, so it can
/// run on another thread while the interpreter executes the segment.
/// With a non-zero `layout.mask` (the arena size minus one, a power of
/// two minus one), guest addresses are masked instead of bounds-checked
/// (`MachineOptions::jit_automatic_nbit_address_space`). Otherwise the
/// code checks them against `layout`'s windows, which are constants in
/// it (C++ `aj_machine_info`'s baked arena bounds); machines whose
/// windows don't contain them interpret the segment instead.
fn compile(exec: &DecodedSegment, layout: MemoryLayout) -> Option<Image> {
    let analysis = region::analyse(exec);
    emit::compile(exec, &analysis, layout)
}

/// Run the JIT code `code` of the current segment from the block starting
/// at `pc` (decoder entry `slot`), with the counter as it was *before*
/// beginning that block.
pub(crate) fn run(
    code: &CodeMemory,
    m: &mut Machine,
    slot: usize,
    pc: u64,
    counter: u64,
    max: u64,
) -> NativeExit {
    let MemoryLayout { read, write, .. } = m.memory.native_layout();
    let frm = u64::from(m.cpu.regs.fcsr().frm());
    let (regs, fregs) = m.cpu.regs.x_f_mut();
    let arena = m.memory.arena_mut();
    let Some(mut state) = JitState::new(regs, fregs, arena, read, write, max, frm) else {
        return NativeExit::Unavailable;
    };
    if !code.enter(&mut state, slot, pc, counter) {
        return NativeExit::Unavailable;
    }
    NativeExit::from_raw(state.exit_kind, state.exit_pc, state.counter)
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
            &[],
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
                    &[],
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
    /// `emit`: the same results as the interpreter.
    #[test]
    #[cfg(all(feature = "ext-f", not(feature = "debug-checks"), target_os = "linux"))]
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
                &[],
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
    static BASELINE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Saved code with libcalls (baseline x86-64: rounding and FMA are
    /// calls) links them against the C library's `nearbyint`, `fma` and so
    /// on, through the GOT: the same results as the interpreter.
    #[test]
    #[cfg(all(feature = "ext-f", not(feature = "debug-checks"), target_os = "linux"))]
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
                &[],
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

    /// A loop head inside a decoder block (straight-line code runs into
    /// it) is a native entry, but its decoder entry isn't patched, since the
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
                &[],
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
}
