//! The Cranelift JIT against the interpreter.
//!
//! The `rvlinux` difftests run every guest with `--jit` against the C++
//! oracle (exit code, output, instruction count). These tests compare the
//! *complete* machine state instead: all registers, PC, the instruction
//! counter, memory and the error, for hand-written and random programs,
//! stopped at many different instruction limits. Without the C extension
//! (`--no-default-features`), only 4-byte instructions are generated.

#![cfg(feature = "jit")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

const V: u64 = 0x10000;
const SCRATCH: u64 = 0x40000;
const SCRATCH_LEN: u64 = 0x2000;
/// The base register of generated loads and stores; never a destination.
const BASE: u32 = 5;

fn machine(code: &[u8], jit: bool) -> Machine {
    machine_with(
        code,
        MachineOptions {
            jit,
            ..Default::default()
        },
    )
}

/// A machine compiling in the background, checking for finished code every
/// `slice` instructions.
fn background_machine(code: &[u8], slice: u64) -> Machine {
    machine_with(
        code,
        MachineOptions {
            jit: true,
            jit_background: true,
            jit_background_slice: slice,
            ..Default::default()
        },
    )
}

fn machine_with(code: &[u8], options: MachineOptions) -> Machine {
    let mut m = Machine::new(
        &[],
        MachineOptions {
            memory_max: 1 << 20,
            ..options
        },
    )
    .unwrap();
    m.set_printer(|_, _| {});
    m.set_rdtime(|_| 1234);
    m.init_execute_area(code, V).unwrap();
    m
}

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    result: Result<bool, (ExceptionKind, u64)>,
    regs: [u64; 32],
    pc: u64,
    counter: u64,
    memory: Vec<u8>,
}

fn run(m: &mut Machine, regs: &[u64; 32], fuel: u64) -> Outcome {
    m.cpu.registers_mut().clone_from(&Default::default());
    for (i, &v) in regs.iter().enumerate().skip(1) {
        m.cpu.set_reg(i as u32, v);
    }
    m.copy_to_guest(SCRATCH, &vec![0x5A; SCRATCH_LEN as usize])
        .unwrap();
    m.cpu.jump(V).unwrap();
    m.set_instruction_counter(0);
    let result = m.try_simulate(fuel, 0).map_err(|e| (e.kind, e.data));
    Outcome {
        result,
        regs: *m.cpu.registers().all(),
        pc: m.cpu.pc(),
        counter: m.instruction_counter(),
        memory: m.memory.memview(SCRATCH, SCRATCH_LEN).unwrap().to_vec(),
    }
}

/// Run `code` with and without the JIT at every limit in `fuels`.
fn compare(code: &[u8], regs: &[u64; 32], fuels: &[u64], what: &str) -> usize {
    compare_machines(machine(code, true), code, regs, fuels, what)
}

/// Run `code` in the interpreter and on `jit` at every limit in `fuels`.
/// Returns the number of decoder entries `jit` enters native code at.
fn compare_machines(
    mut jit: Machine,
    code: &[u8],
    regs: &[u64; 32],
    fuels: &[u64],
    what: &str,
) -> usize {
    let mut interp = machine(code, false);
    let entries = jit.cpu.current_execute_segment().jit_entry_count();
    for &fuel in fuels {
        let expected = run(&mut interp, regs, fuel);
        let got = run(&mut jit, regs, fuel);
        if got != expected {
            let mut diff = String::new();
            if got.result != expected.result {
                diff += &format!(
                    "result: jit {:?}, interpreter {:?}\n",
                    got.result, expected.result
                );
            }
            if got.pc != expected.pc {
                diff += &format!("pc: jit {:#x}, interpreter {:#x}\n", got.pc, expected.pc);
            }
            if got.counter != expected.counter {
                diff += &format!(
                    "counter: jit {}, interpreter {}\n",
                    got.counter, expected.counter
                );
            }
            for r in 0..32 {
                if got.regs[r] != expected.regs[r] {
                    diff += &format!(
                        "x{r}: jit {:#x}, interpreter {:#x}\n",
                        got.regs[r], expected.regs[r]
                    );
                }
            }
            if let Some(i) = (0..got.memory.len()).find(|&i| got.memory[i] != expected.memory[i]) {
                diff += &format!("memory differs from {:#x}\n", SCRATCH + i as u64);
            }
            panic!("{what}, fuel {fuel}:\n{diff}");
        }
    }
    entries
}

fn words(program: &[u32]) -> Vec<u8> {
    program.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn fuels() -> Vec<u64> {
    let mut f: Vec<u64> = (0..=80).collect();
    f.extend([100, 127, 128, 129, 500, 1000, 4096, 10_000, 100_000]);
    f
}

const fn itype(op: u32, f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}
const fn rtype(op: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}
const fn stype(f3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    (((imm >> 5) & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
}
const fn btype(f3: u32, rs1: u32, rs2: u32, offset: i32) -> u32 {
    let o = offset as u32;
    (((o >> 12) & 1) << 31)
        | (((o >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | (((o >> 1) & 0xF) << 8)
        | (((o >> 11) & 1) << 7)
        | 0x63
}
const fn jtype(rd: u32, offset: i32) -> u32 {
    let o = offset as u32;
    (((o >> 20) & 1) << 31)
        | (((o >> 1) & 0x3FF) << 21)
        | (((o >> 11) & 1) << 20)
        | (((o >> 12) & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
}
const ECALL: u32 = 0x73;

#[test]
fn loops_calls_and_returns_stay_native() {
    // sum = 0; for (i = 100; i != 0; i--) sum += f(i), with f(x) = x * 3
    // called through JAL and returning through JALR.
    let program = [
        itype(0x13, 0, 10, 0, 0),      // 0x00: li a0, 0
        itype(0x13, 0, 11, 0, 100),    // 0x04: li a1, 100
        jtype(1, 0x18),                // 0x08: jal ra, f (0x20)
        rtype(0x33, 0, 0, 10, 10, 12), // 0x0C: add a0, a0, a2
        itype(0x13, 0, 11, 11, -1),    // 0x10: addi a1, a1, -1
        btype(1, 11, 0, -12),          // 0x14: bnez a1, 0x08
        jtype(0, 0),                   // 0x18: j .
        0x13,                          // 0x1C: nop
        rtype(0x33, 0, 0, 12, 11, 11), // 0x20: f: add a2, a1, a1
        rtype(0x33, 0, 0, 12, 12, 11), // 0x24: add a2, a2, a1
        itype(0x67, 0, 0, 1, 0),       // 0x28: ret
    ];
    let entries = compare(&words(&program), &[0; 32], &fuels(), "loop");
    assert!(entries > 0, "nothing was compiled");
    let mut m = machine(&words(&program), true);
    run(&mut m, &[0; 32], 1000);
    assert_eq!(m.cpu.reg(10), 3 * 5050);
}

#[test]
fn faults_and_syscalls_leave_the_jit_precisely() {
    let program = [
        itype(0x13, 0, 10, 0, 7),     // li a0, 7
        itype(0x13, 0, 17, 0, 300),   // li a7, 300 (unhandled syscall)
        ECALL,                        // back to the interpreter...
        itype(0x13, 0, 10, 10, 1),    // ...and into native code again
        stype(3, BASE, 10, 8),        // sd a0, 8(t0)
        itype(0x03, 3, 11, BASE, 8),  // ld a1, 8(t0)
        itype(0x03, 2, 12, 0, 0x7F0), // lw a2, 0x7F0(x0): read fault
        jtype(0, 0),
    ];
    let mut regs = [0u64; 32];
    regs[BASE as usize] = SCRATCH;
    compare(&words(&program), &regs, &fuels(), "faults");
    // A store into the read-only window faults too.
    let program = [itype(0x13, 0, 10, 0, 1), stype(0, 0, 10, 0x10), jtype(0, 0)];
    compare(&words(&program), &regs, &fuels(), "write fault");
}

/// Loads and stores outside the arena windows go through the page table:
/// native code hands the instruction (and the rest of its block) to the
/// interpreter, including at the start of a block and across pages.
#[test]
fn accesses_outside_the_arena_use_the_page_table() {
    const FAR: u32 = 6;
    let program = [
        itype(0x13, 0, 10, 0, 0),       // 0x00: li a0, 0
        itype(0x13, 0, 11, 0, 50),      // 0x04: li a1, 50
        itype(0x03, 3, 12, FAR, 0),     // 0x08: loop: ld a2, 0(t1)
        itype(0x13, 0, 12, 12, 3),      // 0x0C: addi a2, a2, 3
        stype(3, FAR, 12, 0),           // 0x10: sd a2, 0(t1)
        stype(3, BASE, 12, 8),          // 0x14: sd a2, 8(t0)
        itype(0x03, 2, 13, FAR, 0xFFE), // 0x18: lw a3, 0xFFE(t1): crosses pages
        stype(1, FAR, 12, 0x7FE),       // 0x1C: sh a2, 0x7FE(t1)
        rtype(0x33, 0, 0, 10, 10, 12),  // 0x20: add a0, a0, a2
        itype(0x13, 0, 11, 11, -1),     // 0x24: addi a1, a1, -1
        btype(1, 11, 0, -32),           // 0x28: bnez a1, loop
        jtype(0, 0),                    // 0x2C: j .
    ];
    let mut regs = [0u64; 32];
    regs[BASE as usize] = SCRATCH;
    regs[FAR as usize] = 0x7000_0000_0802;
    let entries = compare(&words(&program), &regs, &fuels(), "outside the arena");
    assert!(entries > 0, "nothing was compiled");
    let mut m = machine(&words(&program), true);
    let out = run(&mut m, &regs, 10_000);
    assert_eq!(out.result, Ok(false)); // timed out in `j .`
    assert_eq!(m.cpu.reg(10), 3 * (50 * 51 / 2));
    assert_eq!(m.memory.read_u64(0x7000_0000_0802).unwrap(), 150);
    // The page of t1 and the next one, written by the SH.
    assert_eq!(m.memory.owned_pages_active(), 2);
}

#[test]
fn x0_writes_are_discarded() {
    // OP-32 fast paths with rd = x0 are no-ops in the dispatcher (C++
    // rewrites them to `LI x0, 0`); native code must agree.
    let program = [
        itype(0x13, 0, 11, 0, 5),     // li a1, 5
        rtype(0x3B, 0, 0, 0, 11, 11), // addw x0, a1, a1
        rtype(0x33, 0, 0, 10, 0, 0),  // add a0, x0, x0
        btype(0, 0, 0, 8),            // beq x0, x0, +8
        itype(0x13, 0, 10, 10, 1),    // addi a0, a0, 1
        jtype(0, 0),
    ];
    compare(&words(&program), &[0; 32], &fuels(), "x0");
    let mut m = machine(&words(&program), true);
    run(&mut m, &[0; 32], 100);
    assert_eq!((m.cpu.reg(0), m.cpu.reg(10)), (0, 0));
}

/// xorshift64*
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
    /// A destination register: anything but the memory base.
    fn rd(&mut self) -> u32 {
        loop {
            let r = self.below(32) as u32;
            // x0 occasionally, for the rd = x0 paths.
            if r != BASE && (r != 0 || self.below(8) == 0) {
                return r;
            }
        }
    }
    fn rs(&mut self) -> u32 {
        self.below(32) as u32
    }
}

#[derive(Clone, Copy)]
enum Item {
    Word(u32),
    Half(u16),
    /// A branch (funct3, rs1, rs2) to instruction `n`.
    Branch(u32, u32, u32, usize),
    /// JAL rd to instruction `n`.
    Jal(u32, usize),
    /// AUIPC x6 + JALR rd, x6 to instruction `n` (8 bytes).
    Call(u32, usize),
    /// C.BEQZ/C.BNEZ (funct3) on x8+r to instruction `n`.
    CBranch(u16, u32, usize),
    /// C.J to instruction `n`.
    CJump(usize),
}

impl Item {
    const fn len(self) -> u64 {
        match self {
            Self::Half(_) | Self::CBranch(..) | Self::CJump(_) => 2,
            Self::Call(..) => 8,
            _ => 4,
        }
    }
}

fn random_item(rng: &mut Rng, n: usize) -> Item {
    let compressed = libriscv::COMPRESSED_ENABLED;
    let target = |rng: &mut Rng| rng.below(n as u64) as usize;
    match rng.below(if compressed { 20 } else { 16 }) {
        0..=2 => {
            const F7: [u32; 11] = [0, 0, 0x01, 0x20, 0x04, 0x05, 0x07, 0x10, 0x14, 0x24, 0x30];
            let f7 = rng.pick(&F7);
            Item::Word(rtype(
                0x33,
                rng.below(8) as u32,
                f7,
                rng.rd(),
                rng.rs(),
                rng.rs(),
            ))
        }
        3 => {
            const F7: [u32; 6] = [0, 0x01, 0x20, 0x04, 0x10, 0x30];
            let f7 = rng.pick(&F7);
            Item::Word(rtype(
                0x3B,
                rng.below(8) as u32,
                f7,
                rng.rd(),
                rng.rs(),
                rng.rs(),
            ))
        }
        4..=5 => {
            let f3 = rng.below(8) as u32;
            let imm = if f3 == 1 || f3 == 5 {
                const HI: [u32; 7] = [0, 0, 0x400, 0x280, 0x480, 0x680, 0x600];
                const WHOLE: [u32; 8] = [0x600, 0x601, 0x602, 0x604, 0x605, 0x287, 0x6B8, 0x687];
                if rng.below(4) == 0 {
                    rng.pick(&WHOLE)
                } else {
                    rng.pick(&HI) | rng.below(64) as u32
                }
            } else {
                rng.below(4096) as u32
            };
            Item::Word(itype(0x13, f3, rng.rd(), rng.rs(), imm as i32))
        }
        6 => {
            const HI: [u32; 5] = [0, 0x400, 0x080, 0x600, 0x600];
            let f3 = rng.pick(&[0, 1, 5, 1, 5]);
            let imm = if rng.below(4) == 0 {
                rng.pick(&[0x600u32, 0x601, 0x602])
            } else {
                rng.pick(&HI) | rng.below(64) as u32
            };
            Item::Word(itype(0x1B, f3, rng.rd(), rng.rs(), imm as i32))
        }
        7 => {
            let op = rng.pick(&[0x37, 0x17]);
            Item::Word(((rng.next() as u32) & !0xFFF) | (rng.rd() << 7) | op)
        }
        8..=9 => {
            let base = if rng.below(16) == 0 { rng.rs() } else { BASE };
            let off = rng.below(1100) as i32 - 64;
            Item::Word(itype(0x03, rng.below(8) as u32, rng.rd(), base, off))
        }
        10 => {
            let base = if rng.below(16) == 0 { rng.rs() } else { BASE };
            let off = rng.below(1100) as i32 - 64;
            Item::Word(stype(rng.below(4) as u32, base, rng.rs(), off))
        }
        11..=12 => Item::Branch(rng.below(8) as u32, rng.rs(), rng.rs(), target(rng)),
        13 => {
            let other = rng.rd();
            let rd = rng.pick(&[0, 1, 1, other]);
            Item::Jal(rd, target(rng))
        }
        14 => {
            if rng.below(3) == 0 {
                // A random indirect jump, mostly out of the segment.
                Item::Word(itype(0x67, 0, rng.rd(), rng.rs(), rng.below(4096) as i32))
            } else {
                Item::Call(rng.pick(&[0, 1, 1]), target(rng))
            }
        }
        15 => match rng.below(16) {
            0 => Item::Word(ECALL),
            1 => Item::Word(itype(0x13, 0, 17, 0, rng.below(600) as i32)), // li a7, n
            2 => Item::Word(rng.next() as u32 | 3),                        // anything
            3 => Item::Word(0x0000_000F),                                  // FENCE
            _ => Item::Word(itype(0x13, 0, rng.rd(), rng.rs(), rng.below(4096) as i32)),
        },
        16..=17 => {
            // Random compressed instructions (the F/D ones stay interpreted).
            let mut h = rng.next() as u16;
            if h & 3 == 3 {
                h &= !1;
            }
            Item::Half(h)
        }
        18 => Item::CBranch(rng.pick(&[0b110, 0b111]), rng.below(8) as u32, target(rng)),
        _ => Item::CJump(target(rng)),
    }
}

fn cb_offset(o: i32) -> u16 {
    let o = o as u32;
    ((((o >> 8) & 1) << 12)
        | (((o >> 3) & 3) << 10)
        | (((o >> 6) & 3) << 5)
        | (((o >> 1) & 3) << 3)
        | (((o >> 5) & 1) << 2)) as u16
}
fn cj_offset(o: i32) -> u16 {
    let o = o as u32;
    ((((o >> 11) & 1) << 12)
        | (((o >> 4) & 1) << 11)
        | (((o >> 8) & 3) << 9)
        | (((o >> 10) & 1) << 8)
        | (((o >> 6) & 1) << 7)
        | (((o >> 7) & 1) << 6)
        | (((o >> 1) & 7) << 3)
        | (((o >> 5) & 1) << 2)) as u16
}

fn encode(items: &[Item]) -> Vec<u8> {
    let mut pcs = Vec::with_capacity(items.len());
    let mut pc = 0u64;
    for it in items {
        pcs.push(pc);
        pc += it.len();
    }
    let rel = |from: usize, to: usize| (pcs[to] as i64 - pcs[from] as i64) as i32;
    let mut out = Vec::new();
    for (i, it) in items.iter().enumerate() {
        match *it {
            Item::Word(w) => out.extend(w.to_le_bytes()),
            Item::Half(h) => out.extend(h.to_le_bytes()),
            Item::Branch(f3, a, b, t) => out.extend(btype(f3, a, b, rel(i, t)).to_le_bytes()),
            Item::Jal(rd, t) => out.extend(jtype(rd, rel(i, t)).to_le_bytes()),
            Item::Call(rd, t) => {
                out.extend((6 << 7 | 0x17u32).to_le_bytes()); // auipc x6, 0
                out.extend(itype(0x67, 0, rd, 6, rel(i, t)).to_le_bytes());
            }
            Item::CBranch(f3, r, t) => {
                let h = (f3 << 13) | ((r as u16) << 7) | cb_offset(rel(i, t)) | 0b01;
                out.extend(h.to_le_bytes());
            }
            Item::CJump(t) => {
                let h = (0b101 << 13) | cj_offset(rel(i, t)) | 0b01;
                out.extend(h.to_le_bytes());
            }
        }
    }
    out
}

#[test]
fn background_compilation_switches_to_native_code() {
    let program = [
        itype(0x13, 0, 10, 0, 0),      // li a0, 0
        itype(0x13, 0, 11, 0, 100),    // li a1, 100
        rtype(0x33, 0, 0, 10, 10, 11), // add a0, a0, a1
        itype(0x13, 0, 11, 11, -1),    // addi a1, a1, -1
        btype(1, 11, 0, -8),           // bnez a1, -8
        jtype(0, 0),                   // j .
    ];
    let code = words(&program);
    let mut m = background_machine(&code, 3);
    // The segment starts out interpreted.
    assert_eq!(m.cpu.current_execute_segment().jit_entry_count(), 0);
    // Results don't depend on when the code arrives...
    let early = run(&mut m, &[0; 32], 150);
    m.wait_for_jit();
    assert!(!m.jit_pending());
    assert!(m.cpu.current_execute_segment().jit_entry_count() > 0);
    // ...and from here on everything runs natively.
    let late = run(&mut m, &[0; 32], 150);
    assert_eq!(early, late);
    let expected = run(&mut machine(&code, false), &[0; 32], 150);
    assert_eq!(late, expected);
    // Waiting again is a no-op.
    m.wait_for_jit();
}

#[test]
fn background_compilation_matches_the_interpreter() {
    // Wherever the compiled code happens to land (between which slices, at
    // which fuel), every run must match the interpreter.
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    let fuels: Vec<u64> = (0..=48).chain([64, 100, 333, 1000, 20_000]).collect();
    for program in 0..60 {
        let n = 8 + rng.below(56) as usize;
        let items: Vec<Item> = (0..n).map(|_| random_item(&mut rng, n)).collect();
        let code = encode(&items);
        let mut regs = [0u64; 32];
        for r in regs.iter_mut().skip(1) {
            *r = rng.below(64);
        }
        regs[BASE as usize] = SCRATCH + 64;
        let slice = 1 + rng.below(9);
        compare_machines(
            background_machine(&code, slice),
            &code,
            &regs,
            &fuels,
            &format!("program {program} (background, slice {slice}): {code:02x?}"),
        );
    }
}

/// `LIBRISCV_JIT_FUZZ=<programs>[:<seed>]` runs a longer fuzzing session.
#[test]
fn random_programs_match_the_interpreter() {
    let config = std::env::var("LIBRISCV_JIT_FUZZ").unwrap_or_default();
    let mut parts = config.split(':');
    let programs: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(400);
    let seed: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let fuels: Vec<u64> = (0..=48).chain([64, 100, 333, 1000, 20_000]).collect();
    let mut compiled = 0;
    for program in 0..programs {
        let n = 8 + rng.below(56) as usize;
        let items: Vec<Item> = (0..n).map(|_| random_item(&mut rng, n)).collect();
        let code = encode(&items);
        let mut regs = [0u64; 32];
        for r in regs.iter_mut().skip(1) {
            *r = match rng.below(4) {
                0 => rng.below(16),
                1 => rng.next(),
                2 => (rng.next() as i32) as u64,
                _ => [0, 1, u64::MAX, 1 << 63, (1 << 63) - 1, 0x8000_0000][rng.below(6) as usize],
            };
        }
        regs[BASE as usize] = SCRATCH + 64;
        if compare(
            &code,
            &regs,
            &fuels,
            &format!("program {program}: {code:02x?}"),
        ) > 0
        {
            compiled += 1;
        }
    }
    assert!(
        compiled * 4 > programs * 3,
        "only {compiled} of {programs} programs were compiled"
    );
}
