//! Shared by the native-code tests (`jit.rs`, `aot.rs`): machines, the
//! comparison of complete machine states, instruction encoders and the
//! random program generators.

#![allow(dead_code, unreachable_pub)]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

pub const V: u64 = 0x10000;
pub const SCRATCH: u64 = 0x40000;
pub const SCRATCH_LEN: u64 = 0x2000;
/// The base register of generated loads and stores; never a destination.
pub const BASE: u32 = 5;

/// Initial registers: `x0..x31`, then `f0..f31`.
pub type Regs = [u64; 64];

pub fn machine_sized(code: &[u8], memory_max: u64, options: MachineOptions) -> Machine {
    let mut m = Machine::new(
        &[],
        MachineOptions {
            memory_max,
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
pub struct Outcome {
    pub result: Result<bool, (ExceptionKind, u64)>,
    pub regs: [u64; 32],
    pub fregs: [u64; 32],
    pub pc: u64,
    pub counter: u64,
    pub memory: Vec<u8>,
}

pub fn run(m: &mut Machine, regs: &Regs, fuel: u64) -> Outcome {
    m.cpu.registers_mut().clone_from(&Default::default());
    for (i, &v) in regs[..32].iter().enumerate().skip(1) {
        m.cpu.set_reg(i as u32, v);
    }
    for (i, &v) in regs[32..].iter().enumerate() {
        m.cpu.registers_mut().set_fp_bits(i as u32, v);
    }
    m.copy_to_guest(SCRATCH, &vec![0x5A; SCRATCH_LEN as usize])
        .unwrap();
    m.cpu.jump(V).unwrap();
    m.set_instruction_counter(0);
    let result = m.try_simulate(fuel, 0).map_err(|e| (e.kind, e.data));
    Outcome {
        result,
        regs: *m.cpu.registers().all(),
        fregs: *m.cpu.registers().all_fp(),
        pc: m.cpu.pc(),
        counter: m.instruction_counter(),
        memory: m.memory.memview(SCRATCH, SCRATCH_LEN).unwrap().to_vec(),
    }
}

/// A NaN as either a double or (in the low half) a single.
pub fn nanish(bits: u64) -> bool {
    f64::from_bits(bits).is_nan() || f32::from_bits(bits as u32).is_nan()
}

/// Equal, or NaN in the same places.
pub fn same_fp(a: u64, b: u64) -> bool {
    a == b
        || (f64::from_bits(a).is_nan() && f64::from_bits(b).is_nan())
        || (f32::from_bits(a as u32).is_nan()
            && f32::from_bits(b as u32).is_nan()
            && a >> 32 == b >> 32)
}

pub fn describe(got: &Outcome, expected: &Outcome) -> String {
    let mut diff = String::new();
    if got.result != expected.result {
        diff += &format!(
            "result: native {:?}, interpreter {:?}\n",
            got.result, expected.result
        );
    }
    if got.pc != expected.pc {
        diff += &format!("pc: native {:#x}, interpreter {:#x}\n", got.pc, expected.pc);
    }
    if got.counter != expected.counter {
        diff += &format!(
            "counter: native {}, interpreter {}\n",
            got.counter, expected.counter
        );
    }
    for r in 0..32 {
        if got.regs[r] != expected.regs[r] {
            diff += &format!(
                "x{r}: native {:#x}, interpreter {:#x}\n",
                got.regs[r], expected.regs[r]
            );
        }
        if got.fregs[r] != expected.fregs[r] {
            diff += &format!(
                "f{r}: native {:#x}, interpreter {:#x}\n",
                got.fregs[r], expected.fregs[r]
            );
        }
    }
    if let Some(i) = (0..got.memory.len()).find(|&i| got.memory[i] != expected.memory[i]) {
        diff += &format!("memory differs from {:#x}\n", SCRATCH + i as u64);
    }
    diff
}

pub fn words(program: &[u32]) -> Vec<u8> {
    program.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub fn fuels() -> Vec<u64> {
    let mut f: Vec<u64> = (0..=80).collect();
    f.extend([100, 127, 128, 129, 500, 1000, 4096, 10_000, 100_000]);
    f
}

pub const fn itype(op: u32, f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}
pub const fn rtype(op: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}
pub const fn stype(f3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    (((imm >> 5) & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
}
pub const fn btype(f3: u32, rs1: u32, rs2: u32, offset: i32) -> u32 {
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
pub const fn jtype(rd: u32, offset: i32) -> u32 {
    let o = offset as u32;
    (((o >> 20) & 1) << 31)
        | (((o >> 1) & 0x3FF) << 21)
        | (((o >> 11) & 1) << 20)
        | (((o >> 12) & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
}
pub const ECALL: u32 = 0x73;

/// A minimal static ELF: one read+execute `PT_LOAD` holding `code` at `V`,
/// followed by `pad` zero bytes, so that read-only data ends past them and
/// the write window starts there (inside the read window, unlike a machine
/// without a program, whose write window starts at 0). The loader wants a
/// section table: one empty string table, after the segment.
pub fn elf_with(code: &[u8], pad: usize) -> Vec<u8> {
    // The segment starts with the code, so that it begins a block.
    const HDR: u64 = 0x80;
    let len = code.len() as u64 + pad as u64;
    let mut elf = Vec::new();
    elf.extend_from_slice(&[0x7F, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_type: ET_EXEC
    elf.extend_from_slice(&243u16.to_le_bytes()); // e_machine: EM_RISCV
    elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    elf.extend_from_slice(&V.to_le_bytes()); // e_entry
    elf.extend_from_slice(&64u64.to_le_bytes()); // e_phoff
    elf.extend_from_slice(&(HDR + len).to_le_bytes()); // e_shoff
    elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    for v in [64u16, 56, 1, 64, 1, 0] {
        // e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx
        elf.extend_from_slice(&v.to_le_bytes());
    }
    elf.extend_from_slice(&1u32.to_le_bytes()); // p_type: PT_LOAD
    elf.extend_from_slice(&5u32.to_le_bytes()); // p_flags: R | X
    for v in [HDR, V, V, len, len, 0x1000u64] {
        // p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align
        elf.extend_from_slice(&v.to_le_bytes());
    }
    elf.resize(HDR as usize, 0);
    elf.extend_from_slice(code);
    elf.resize((HDR + len) as usize, 0);
    // sh_name, sh_type (SHT_STRTAB): its name is the NUL at offset HDR - 1.
    elf.extend_from_slice(&0u32.to_le_bytes());
    elf.extend_from_slice(&3u32.to_le_bytes());
    for v in [0, 0, HDR - 1, 1u64] {
        // sh_flags, sh_addr, sh_offset, sh_size
        elf.extend_from_slice(&v.to_le_bytes());
    }
    elf.extend_from_slice(&[0; 24]); // sh_link, sh_info, sh_addralign, sh_entsize
    elf
}

/// xorshift64*
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
    /// A destination register: anything but the memory base.
    pub fn rd(&mut self) -> u32 {
        loop {
            let r = self.below(32) as u32;
            // x0 occasionally, for the rd = x0 paths.
            if r != BASE && (r != 0 || self.below(8) == 0) {
                return r;
            }
        }
    }
    pub fn rs(&mut self) -> u32 {
        self.below(32) as u32
    }
}

#[derive(Clone, Copy)]
pub enum Item {
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
    pub const fn len(self) -> u64 {
        match self {
            Self::Half(_) | Self::CBranch(..) | Self::CJump(_) => 2,
            Self::Call(..) => 8,
            _ => 4,
        }
    }
}

pub fn random_item(rng: &mut Rng, n: usize) -> Item {
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
            4..=6 if cfg!(feature = "ext-f") && !cfg!(feature = "debug-checks") => {
                Item::Word(random_fp_move(rng))
            }
            _ => Item::Word(itype(0x13, 0, rng.rd(), rng.rs(), rng.below(4096) as i32)),
        },
        16..=17 => {
            // Random compressed instructions.
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

pub fn cb_offset(o: i32) -> u16 {
    let o = o as u32;
    ((((o >> 8) & 1) << 12)
        | (((o >> 3) & 3) << 10)
        | (((o >> 6) & 3) << 5)
        | (((o >> 1) & 3) << 3)
        | (((o >> 5) & 1) << 2)) as u16
}
pub fn cj_offset(o: i32) -> u16 {
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

pub fn encode(items: &[Item]) -> Vec<u8> {
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

// --- F and D ---------------------------------------------------------------

/// OP-FP: `rd = rs1 op rs2` with funct7 = `fpfunc << 2 | fmt`.
pub const fn fp(fpfunc: u32, fmt: u32, rm: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0x53, rm, (fpfunc << 2) | fmt, rd, rs1, rs2)
}
/// An R4-type fused multiply-add (`op` is the major opcode).
pub const fn r4(op: u32, fmt: u32, rm: u32, rd: u32, rs1: u32, rs2: u32, rs3: u32) -> u32 {
    (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15) | (rm << 12) | (rd << 7) | op
}
/// FLW (width 2) / FLD (width 3).
pub const fn fload(width: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(0x07, width, rd, rs1, imm)
}
/// FSW (width 2) / FSD (width 3).
pub const fn fstore(width: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    stype(width, rs1, rs2, imm) | 0x04
}
/// `fsrmi x0, rm`: CSRRWI on `frm`.
pub const fn fsrmi(rm: u32) -> u32 {
    (0x002 << 20) | (rm << 15) | (5 << 12) | 0x73
}

pub const SPECIAL_F64: [f64; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    1.5,
    2.5,
    -2.5,
    3.5,
    -3.5,
    2147483647.5,
    2147483648.0,
    -2147483648.5,
    4294967295.5,
    4294967296.0,
    9.3e18,
    -9.3e18,
    1.9e19,
    1e300,
    f64::INFINITY,
    f64::NEG_INFINITY,
    5e-324,
    -1e-310,
];

/// A random non-NaN FP register value: a double, or a single in the low
/// half.
pub fn random_fp(rng: &mut Rng) -> u64 {
    loop {
        let bits = match rng.below(5) {
            0 => rng.pick(&SPECIAL_F64).to_bits(),
            1 => ((rng.next() as i64 % 4_000_000) as f64 / 8.0).to_bits(),
            2 | 3 => {
                let v = if rng.below(2) == 0 {
                    rng.pick(&SPECIAL_F64) as f32
                } else {
                    (rng.next() as i32 % 1_000_000) as f32 / 4.0
                };
                let random = rng.next() & 0x3FFF_FFFF;
                let upper = rng.pick(&[0, 0x3FF0_0000, random]);
                (upper << 32) | u64::from(v.to_bits())
            }
            _ => rng.next(),
        };
        if !nanish(bits) {
            return bits;
        }
    }
}

/// An F/D instruction whose result doesn't depend on which NaN the host
/// produces: loads, stores, moves, sign injection, FCLASS, compares and
/// conversions to and from integers.
pub fn random_fp_move(rng: &mut Rng) -> u32 {
    let fmt = rng.below(2) as u32;
    let (frd, frs1, frs2) = (rng.rs(), rng.rs(), rng.rs());
    let off = rng.below(1100) as i32 - 64;
    match rng.below(9) {
        0 => fload(2 + fmt, frd, BASE, off),
        1 => fstore(2 + fmt, BASE, frs2, off),
        2 => fp(0x1C, fmt, rng.below(2) as u32, rng.rd(), frs1, 0), // FMV.X, FCLASS
        3 => fp(0x1E, fmt, 0, frd, rng.rs(), 0),                    // FMV.W.X/D.X
        4 => fp(0x04, fmt, rng.below(4) as u32, frd, frs1, frs2),   // FSGNJ*
        5 => fp(0x14, fmt, rng.below(3) as u32, rng.rd(), frs1, frs2), // FLE/FLT/FEQ
        6 => fp(
            0x18,
            fmt,
            rng.below(8) as u32,
            rng.rd(),
            frs1,
            rng.below(4) as u32,
        ),
        7 => fp(
            0x1A,
            fmt,
            rng.below(8) as u32,
            frd,
            rng.rs(),
            rng.below(4) as u32,
        ),
        _ => rng.next() as u32 & !0x7F | rng.pick(&[0x07, 0x27, 0x43, 0x53]),
    }
}

/// Any F/D instruction, mostly on `f0..f7`, and the occasional rounding
/// mode change or integer operation.
pub fn random_fp_item(rng: &mut Rng) -> u32 {
    let f = |rng: &mut Rng| rng.below(8) as u32;
    let fmt = if rng.below(32) == 0 {
        2
    } else {
        rng.below(2) as u32
    };
    let rm = rng.below(8) as u32;
    let (frd, frs1, frs2) = (f(rng), f(rng), f(rng));
    match rng.below(16) {
        0..=2 => fp(rng.below(4) as u32, fmt, rm, frd, frs1, frs2),
        3 => fp(0x05, fmt, rng.pick(&[0, 1, 0, 1, 2]), frd, frs1, frs2), // FMIN/FMAX
        4 => fp(0x0B, fmt, rm, frd, frs1, rng.below(2) as u32),          // FSQRT
        5..=6 => r4(
            rng.pick(&[0x43, 0x47, 0x4B, 0x4F]),
            fmt,
            rm,
            frd,
            frs1,
            frs2,
            f(rng),
        ),
        7 => {
            let src = if rng.below(16) == 0 {
                rng.pick(&[2, 4])
            } else {
                fmt ^ 1
            };
            fp(0x08, fmt, rm, frd, frs1, src) // FCVT.S.D / FCVT.D.S
        }
        8..=9 => {
            let kind = if rng.below(16) == 0 {
                8
            } else {
                rng.below(4) as u32
            };
            fp(0x18, fmt, rm, rng.rd(), frs1, kind) // FCVT.int.fmt
        }
        10 => fsrmi(rng.below(8) as u32),
        11 => itype(0x13, 0, rng.rd(), rng.rs(), rng.below(4096) as i32 - 2048),
        _ => {
            // The same as in the integer programs, on f0..f7.
            let w = random_fp_move(rng);
            let reg_mask = !((0x18 << 7) | (0x18 << 15) | (0x18 << 20));
            if w & 0x7F == 0x53 && (w >> 27) != 0x1C && (w >> 27) != 0x14 && (w >> 27) != 0x18 {
                w & reg_mask
            } else {
                w
            }
        }
    }
}

/// `auipc rd, 0`.
pub const fn auipc(rd: u32) -> u32 {
    (rd << 7) | 0x17
}

/// What [`native_call_programs`] leave in `a0` when they run to the end,
/// where it says something.
pub const NATIVE_CALL_RESULTS: [Option<u64>; 5] = [Some(5050), Some(1275), None, Some(5 + 7), None];

/// Programs for native calls (guest calls as host calls, returning to the
/// caller's region): recursion past the native depth limit, also leaving
/// for the host at its deepest, `auipc`+`jalr` and indirect calls, a
/// callee writing a base register the caller checks on entry, a callee
/// leaving for the host (the caller's frame is gone: its return goes
/// through the entry table), a return elsewhere than after the call, and
/// FP registers live across calls. With their initial registers.
pub fn native_call_programs() -> Vec<(Vec<u8>, Regs)> {
    let mut programs = Vec::new();
    // sum(n) = n + sum(n - 1), recursively, 100 deep.
    let mut stack_regs = [0u64; 64];
    stack_regs[2] = SCRATCH + SCRATCH_LEN;
    programs.push((
        words(&[
            itype(0x13, 0, 10, 0, 100),    // 0x00: li a0, 100
            jtype(1, 0x0C),                // 0x04: jal ra, f (0x10)
            jtype(0, 0),                   // 0x08: j .
            0x13,                          // 0x0C: nop
            itype(0x13, 0, 2, 2, -16),     // 0x10: f: addi sp, sp, -16
            stype(3, 2, 1, 8),             // 0x14: sd ra, 8(sp)
            stype(3, 2, 10, 0),            // 0x18: sd a0, 0(sp)
            btype(0, 10, 0, 0x18),         // 0x1C: beqz a0, 0x34
            itype(0x13, 0, 10, 10, -1),    // 0x20: addi a0, a0, -1
            jtype(1, -0x14),               // 0x24: jal ra, f
            itype(0x03, 3, 11, 2, 0),      // 0x28: ld a1, 0(sp)
            rtype(0x33, 0, 0, 10, 10, 11), // 0x2C: add a0, a0, a1
            jtype(0, 8),                   // 0x30: j 0x38
            itype(0x13, 0, 10, 0, 0),      // 0x34: li a0, 0
            itype(0x03, 3, 1, 2, 8),       // 0x38: ld ra, 8(sp)
            itype(0x13, 0, 2, 2, 16),      // 0x3C: addi sp, sp, 16
            itype(0x67, 0, 0, 1, 0),       // 0x40: ret
        ]),
        stack_regs,
    ));
    // `auipc`+`jalr` to g, an indirect call to h, which moves t0, the
    // base register of the caller's loads.
    let mut base_regs = [0u64; 64];
    base_regs[BASE as usize] = SCRATCH;
    programs.push((
        words(&[
            itype(0x13, 0, 10, 0, 0),      // 0x00: li a0, 0
            itype(0x13, 0, 11, 0, 10),     // 0x04: li a1, 10
            itype(0x03, 3, 13, BASE, 8),   // 0x08: loop: ld a3, 8(t0)
            auipc(1),                      // 0x0C: auipc ra, 0
            itype(0x67, 0, 1, 1, 0x30),    // 0x10: jalr ra, 0x30(ra): g
            auipc(6),                      // 0x14: auipc t1, 0
            itype(0x13, 0, 6, 6, 0x30),    // 0x18: addi t1, t1, 0x30: h
            itype(0x67, 0, 1, 6, 0),       // 0x1C: jalr ra, 0(t1)
            itype(0x03, 3, 12, BASE, 0),   // 0x20: ld a2, 0(t0)
            rtype(0x33, 0, 0, 10, 10, 12), // 0x24: add a0, a0, a2
            rtype(0x33, 0, 0, 10, 10, 13), // 0x28: add a0, a0, a3
            itype(0x13, 0, 11, 11, -1),    // 0x2C: addi a1, a1, -1
            btype(1, 11, 0, -0x28),        // 0x30: bnez a1, loop
            jtype(0, 0),                   // 0x34: j .
            0x13,                          // 0x38: nop
            itype(0x13, 0, 10, 10, 1),     // 0x3C: g: addi a0, a0, 1
            itype(0x67, 0, 0, 1, 0),       // 0x40: ret
            itype(0x13, 0, BASE, BASE, 8), // 0x44: h: addi t0, t0, 8
            stype(3, BASE, 11, 0),         // 0x48: sd a1, 0(t0)
            itype(0x67, 0, 0, 1, 0),       // 0x4C: ret
        ]),
        base_regs,
    ));
    // f makes a system call (native code leaves for the host), k returns
    // past the instruction after its call.
    programs.push((
        words(&[
            itype(0x13, 0, 11, 0, 5),    // 0x00: li a1, 5
            jtype(1, 0x1C),              // 0x04: loop: jal ra, f (0x20)
            itype(0x13, 0, 11, 11, -1),  // 0x08: addi a1, a1, -1
            btype(1, 11, 0, -8),         // 0x0C: bnez a1, loop
            jtype(1, 0x20),              // 0x10: jal ra, k (0x30)
            itype(0x13, 0, 10, 10, 100), // 0x14: addi a0, a0, 100 (skipped)
            itype(0x13, 0, 10, 10, 7),   // 0x18: addi a0, a0, 7
            jtype(0, 0),                 // 0x1C: j .
            itype(0x13, 0, 17, 0, 300),  // 0x20: f: li a7, 300 (unhandled)
            ECALL,                       // 0x24: ecall
            itype(0x13, 0, 10, 10, 1),   // 0x28: addi a0, a0, 1
            itype(0x67, 0, 0, 1, 0),     // 0x2C: ret
            itype(0x13, 0, 1, 1, 4),     // 0x30: k: addi ra, ra, 4
            itype(0x67, 0, 0, 1, 0),     // 0x34: ret
        ]),
        [0; 64],
    ));
    // FP registers live across calls: f0 doubles in g.
    #[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
    programs.push((
        words(&[
            itype(0x13, 0, 10, 0, 3),   // 0x00: li a0, 3
            fp(0x1E, 1, 0, 0, 10, 0),   // 0x04: fmv.d.x f0, a0
            itype(0x13, 0, 11, 0, 4),   // 0x08: li a1, 4
            jtype(1, 0x14),             // 0x0C: loop: jal ra, g (0x20)
            itype(0x13, 0, 11, 11, -1), // 0x10: addi a1, a1, -1
            btype(1, 11, 0, -8),        // 0x14: bnez a1, loop
            fp(0x1C, 1, 0, 12, 0, 0),   // 0x18: fmv.x.d a2, f0
            jtype(0, 0),                // 0x1C: j .
            fp(0x00, 1, 0, 0, 0, 0),    // 0x20: g: fadd.d f0, f0, f0
            itype(0x67, 0, 0, 1, 0),    // 0x24: ret
        ]),
        [0; 64],
    ));
    // Recursion 50 deep that makes a system call at the bottom: native
    // code leaves for the host from inside every native caller.
    programs.insert(
        1,
        (
            words(&[
                itype(0x13, 0, 10, 0, 50),     // 0x00: li a0, 50
                jtype(1, 0x0C),                // 0x04: jal ra, f (0x10)
                jtype(0, 0),                   // 0x08: j .
                0x13,                          // 0x0C: nop
                itype(0x13, 0, 2, 2, -16),     // 0x10: f: addi sp, sp, -16
                stype(3, 2, 1, 8),             // 0x14: sd ra, 8(sp)
                stype(3, 2, 10, 0),            // 0x18: sd a0, 0(sp)
                btype(0, 10, 0, 0x18),         // 0x1C: beqz a0, 0x34
                itype(0x13, 0, 10, 10, -1),    // 0x20: addi a0, a0, -1
                jtype(1, -0x14),               // 0x24: jal ra, f
                itype(0x03, 3, 11, 2, 0),      // 0x28: ld a1, 0(sp)
                rtype(0x33, 0, 0, 10, 10, 11), // 0x2C: add a0, a0, a1
                jtype(0, 0x10),                // 0x30: j 0x40
                itype(0x13, 0, 17, 0, 300),    // 0x34: li a7, 300 (unhandled)
                ECALL,                         // 0x38: ecall
                itype(0x13, 0, 10, 0, 0),      // 0x3C: li a0, 0
                itype(0x03, 3, 1, 2, 8),       // 0x40: ld ra, 8(sp)
                itype(0x13, 0, 2, 2, 16),      // 0x44: addi sp, sp, 16
                itype(0x67, 0, 0, 1, 0),       // 0x48: ret
            ]),
            programs[0].1,
        ),
    );
    programs
}
