//! Instruction encoders for the harness tests.

pub const ECALL: u32 = 0x73;
pub const RET: u32 = 0x8067; // jalr x0, 0(ra)

pub fn words(program: &[u32]) -> Vec<u8> {
    program.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    ((imm as u32 & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
}

/// `lui rd, addr` for a page-aligned address below 2 GiB.
pub const fn lui(rd: u32, addr: u64) -> u32 {
    (addr as u32 & 0xffff_f000) | (rd << 7) | 0x37
}

pub const fn sd(rs2: u32, rs1: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    ((imm >> 5 & 0x7f) << 25) | (rs2 << 20) | (rs1 << 15) | (3 << 12) | ((imm & 31) << 7) | 0x23
}

pub const fn jal(rd: u32, offset: i32) -> u32 {
    let o = offset as u32;
    ((o >> 20 & 1) << 31)
        | ((o >> 1 & 0x3ff) << 21)
        | ((o >> 11 & 1) << 20)
        | ((o >> 12 & 0xff) << 12)
        | (rd << 7)
        | 0x6f
}
