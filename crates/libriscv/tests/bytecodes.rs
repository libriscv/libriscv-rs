//! The fast-path bytecodes (`bytecode.rs`: re-encoded operands, one handler
//! per C++ bytecode) against the precise interpreter, on random programs of
//! non-control-flow RV64IMFDC + B instructions: both must end with the same
//! registers, the same memory and the same fault.
//!
//! There is no C++ counterpart; C++ relies on its fuzzers for this.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineError, MachineOptions};

/// Where the program goes.
const CODE: u64 = 0x2000;
/// The memory that loads and stores mostly hit; base registers point into
/// its middle, so a 12-bit offset stays inside.
const DATA: u64 = 0x10000;
const DATA_LEN: u64 = 0x2000;
const INSTRUCTIONS: usize = 24;
const PROGRAMS: usize = 4000;

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
    /// Mostly registers holding data pointers (x8..x15 and sp), sometimes
    /// any register, x0 included.
    fn reg(&mut self) -> u32 {
        if self.below(4) == 0 {
            self.below(32) as u32
        } else {
            self.pick(&[2, 8, 9, 10, 11, 12, 13, 14, 15])
        }
    }
}

const fn rtype(opcode: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode
}
const fn itype(opcode: u32, f3: u32, rd: u32, rs1: u32, imm: u32) -> u32 {
    ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode
}
const fn stype(opcode: u32, f3: u32, rs1: u32, rs2: u32, imm: u32) -> u32 {
    (((imm >> 5) & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | ((imm & 0x1F) << 7)
        | opcode
}

/// One random 32-bit instruction that doesn't change control flow.
fn long_instruction(r: &mut Rng) -> u32 {
    let (rd, rs1, rs2) = (r.reg(), r.reg(), r.reg());
    let imm = r.next() as u32;
    let shamt = r.below(64) as u32;
    match r.below(12) {
        0 => itype(0x03, r.below(7) as u32, rd, rs1, imm), // LOAD
        1 => stype(0x23, r.below(4) as u32, rs1, rs2, imm), // STORE
        2 => {
            // OP-IMM, with valid shift and bit-manipulation encodings
            let f3 = r.below(8) as u32;
            let imm = match f3 {
                1 => r.pick(&[0x000, 0x280, 0x600, 0x601, 0x602, 0x604, 0x605]) | shamt,
                5 => r.pick(&[0x000, 0x400, 0x480, 0x600, 0x287, 0x698]) | shamt,
                _ => imm,
            };
            itype(0x13, f3, rd, if r.below(8) == 0 { 0 } else { rs1 }, imm)
        }
        3 => {
            // OP, including M, Zba, Zbb, Zbs, Zicond
            let f7 = r.pick(&[
                0x00, 0x20, 0x01, 0x10, 0x05, 0x04, 0x30, 0x24, 0x14, 0x34, 0x07,
            ]);
            rtype(0x33, r.below(8) as u32, f7, rd, rs1, rs2)
        }
        4 => {
            // OP-IMM-32
            let f3 = r.pick(&[0, 1, 5, 1, 5]);
            let hi = r.pick(&[0x000, 0x400, 0x080, 0x600, 0x020]);
            itype(
                0x1B,
                f3,
                rd,
                rs1,
                if f3 == 0 { imm } else { hi | (shamt & 31) },
            )
        }
        5 => {
            // OP-32
            let f7 = r.pick(&[0x00, 0x20, 0x01, 0x04, 0x10, 0x30]);
            let rs2 = if r.below(4) == 0 { 0 } else { rs2 };
            rtype(0x3B, r.below(8) as u32, f7, rd, rs1, rs2)
        }
        6 => (imm & 0xFFFF_F000) | (rd << 7) | r.pick(&[0x37, 0x17]), // LUI, AUIPC
        7 => itype(0x07, r.pick(&[2, 3]), rd, rs1, imm),              // FLW, FLD
        8 => stype(0x27, r.pick(&[2, 3]), rs1, rs2, imm),             // FSW, FSD
        9 => {
            // FADD/FSUB/FMUL/FDIV.S/D
            let f7 = (r.below(4) as u32) << 2 | r.below(2) as u32;
            rtype(0x53, r.pick(&[0, 1, 2, 3, 4, 7]), f7, rd, rs1, rs2)
        }
        10 => {
            // FMADD.S/D
            let rs3 = r.below(32) as u32;
            rtype(0x43, 0, (rs3 << 2) | r.below(2) as u32, rd, rs1, rs2)
        }
        _ => itype(0x13, 0, rd, rs1, imm), // more ADDI/LI/MV
    }
}

/// One random 16-bit instruction that doesn't change control flow.
fn short_instruction(r: &mut Rng) -> u16 {
    loop {
        let c = r.next() as u16;
        let quadrant = c & 3;
        let funct3 = c >> 13;
        let rs2 = (c >> 2) & 31;
        let control = match quadrant {
            // C.JAL/C.J, C.BEQZ, C.BNEZ (C.JAL is C.ADDIW on RV64)
            1 => matches!(funct3, 0b101..=0b111),
            // C.JR, C.JALR, C.EBREAK
            2 => funct3 == 0b100 && rs2 == 0,
            _ => true,
        };
        if !control {
            return c;
        }
    }
}

fn program(r: &mut Rng) -> Vec<u8> {
    let mut code = Vec::new();
    for _ in 0..INSTRUCTIONS {
        if r.below(2) == 0 {
            code.extend(long_instruction(r).to_le_bytes());
        } else {
            code.extend(short_instruction(r).to_le_bytes());
        }
    }
    code
}

/// Fresh registers: data pointers, small and large values.
fn setup(m: &mut Machine, r: &mut Rng) {
    let regs = m.cpu.registers_mut();
    *regs = Default::default();
    for i in 1..32 {
        let v = match r.below(5) {
            0 | 1 => DATA + DATA_LEN / 2 + r.below(64) * 8,
            2 => r.below(64),
            3 => r.next(),
            _ => DATA + DATA_LEN / 2 + r.below(4096),
        };
        regs.set(i, v);
        regs.set_fp_bits(i, r.next());
    }
}

struct Outcome {
    x: Vec<u64>,
    f: Vec<u64>,
    data: Vec<u8>,
    error: Option<MachineError>,
}

fn run(code: &[u8], seed: u64, precise: bool) -> Outcome {
    let mut m = Machine::new(&[], MachineOptions::default()).unwrap();
    // The program ends in `j .`, so the dispatcher runs until its limit.
    let mut code = code.to_vec();
    code.extend(0x0000_006Fu32.to_le_bytes());
    m.init_execute_area(&code, CODE).unwrap();
    let mut r = Rng(seed);
    for a in (DATA..DATA + DATA_LEN).step_by(8) {
        m.memory.write_u64(a, r.next()).unwrap();
    }
    setup(&mut m, &mut r);
    m.cpu.jump(CODE).unwrap();
    let error = if precise {
        m.set_instruction_counter(0);
        m.set_max_instructions(INSTRUCTIONS as u64);
        m.simulate_precise().err()
    } else {
        m.try_simulate(INSTRUCTIONS as u64 + 100, 0).err()
    };
    let regs = m.cpu.registers();
    Outcome {
        x: (0..32).map(|i| regs.get(i)).collect(),
        f: (0..32).map(|i| regs.fp_bits(i)).collect(),
        data: (DATA..DATA + DATA_LEN)
            .map(|a| m.memory.read_u8(a).unwrap())
            .collect(),
        error,
    }
}

#[test]
fn bytecodes_match_the_precise_interpreter() {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    for n in 0..PROGRAMS {
        let code = program(&mut r);
        let seed = r.next() | 1;
        let fast = run(&code, seed, false);
        let precise = run(&code, seed, true);
        let context = format!("program {n}: {code:02x?}");
        // The two interpreters report different data for illegal
        // instructions (the dispatcher's is the decoder entry's bits, as in
        // C++), so only the kind is compared for those.
        let fault = |o: &Outcome| {
            o.error.as_ref().map(|e| match e.kind {
                ExceptionKind::IllegalOpcode | ExceptionKind::UnimplementedInstruction => {
                    (e.kind, 0)
                }
                kind => (kind, e.data),
            })
        };
        assert_eq!(fault(&fast), fault(&precise), "{context}");
        assert_eq!(fast.x, precise.x, "{context}");
        assert_eq!(fast.f, precise.f, "{context}");
        assert!(fast.data == precise.data, "memory differs, {context}");
    }
}
