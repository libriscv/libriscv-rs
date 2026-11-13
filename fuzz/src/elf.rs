//! A minimal RV64 executable around fuzz input, for the harnesses whose
//! setup needs a real ELF: `setup_linux` pushes the program headers, and
//! the libc fast path finds its functions through `.symtab`.
//!
//! Layout: the input is one read+execute `PT_LOAD` at [`CODE`] (the entry
//! point), followed by a zero-filled read+write `PT_LOAD` at [`DATA`].
//! A section table with `.symtab`, `.strtab` and `.shstrtab` follows the
//! code in the file (the loader rejects an ELF without sections).

/// Address of the code (the fuzz input) and the entry point.
pub const CODE: u64 = 0x10000;
/// Address of the zero-filled read-write data segment.
pub const DATA: u64 = 0x20_0000;
/// Size of the data segment.
pub const DATA_LEN: u64 = 0x1_0000;

const EHDR: usize = 64;
const PHDR: usize = 56;
const SHDR: usize = 64;
const SYM: usize = 24;
/// File offset of the code.
const CODE_OFFSET: usize = 0x1000;

const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;
const STT_FUNC: u8 = 2;
const STB_GLOBAL: u8 = 1;

/// A function symbol: name and offset into the code.
pub struct Symbol<'a> {
    pub name: &'a str,
    pub offset: u64,
}

fn put(out: &mut [u8], at: usize, bytes: &[u8]) {
    if let Some(dst) = out.get_mut(at..at + bytes.len()) {
        dst.copy_from_slice(bytes);
    }
}

fn phdr(out: &mut [u8], at: usize, flags: u32, offset: u64, vaddr: u64, filesz: u64, memsz: u64) {
    put(out, at, &1u32.to_le_bytes()); // PT_LOAD
    put(out, at + 4, &flags.to_le_bytes());
    put(out, at + 8, &offset.to_le_bytes());
    put(out, at + 16, &vaddr.to_le_bytes());
    put(out, at + 24, &vaddr.to_le_bytes());
    put(out, at + 32, &filesz.to_le_bytes());
    put(out, at + 40, &memsz.to_le_bytes());
    put(out, at + 48, &0x1000u64.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
fn shdr(
    out: &mut Vec<u8>,
    name: u32,
    sh_type: u32,
    offset: usize,
    size: usize,
    link: u32,
    entsize: u64,
) {
    let mut h = [0u8; SHDR];
    put(&mut h, 0, &name.to_le_bytes());
    put(&mut h, 4, &sh_type.to_le_bytes());
    put(&mut h, 24, &(offset as u64).to_le_bytes());
    put(&mut h, 32, &(size as u64).to_le_bytes());
    put(&mut h, 40, &link.to_le_bytes());
    put(&mut h, 56, &entsize.to_le_bytes());
    out.extend_from_slice(&h);
}

/// An executable running `code` at [`CODE`], with `symbols` (if any) as
/// `STT_FUNC` symbols relative to [`CODE`].
pub fn executable(code: &[u8], symbols: &[Symbol<'_>]) -> Vec<u8> {
    let mut out = vec![0u8; CODE_OFFSET];
    out.extend_from_slice(code);

    // ELF header
    put(&mut out, 0, b"\x7fELF\x02\x01\x01");
    put(&mut out, 16, &2u16.to_le_bytes()); // ET_EXEC
    put(&mut out, 18, &243u16.to_le_bytes()); // EM_RISCV
    put(&mut out, 20, &1u32.to_le_bytes()); // EV_CURRENT
    put(&mut out, 24, &CODE.to_le_bytes()); // e_entry
    put(&mut out, 32, &(EHDR as u64).to_le_bytes()); // e_phoff
    put(&mut out, 52, &(EHDR as u16).to_le_bytes()); // e_ehsize
    put(&mut out, 54, &(PHDR as u16).to_le_bytes()); // e_phentsize
    put(&mut out, 56, &2u16.to_le_bytes()); // e_phnum
    put(&mut out, 58, &(SHDR as u16).to_le_bytes()); // e_shentsize

    let len = code.len() as u64;
    phdr(
        &mut out,
        EHDR,
        PF_R | PF_X,
        CODE_OFFSET as u64,
        CODE,
        len,
        len,
    );
    phdr(&mut out, EHDR + PHDR, PF_R | PF_W, 0, DATA, 0, DATA_LEN);

    // .strtab
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let strtab = out.len();
    out.push(0);
    let mut names = Vec::with_capacity(symbols.len());
    for s in symbols {
        names.push((out.len() - strtab) as u32);
        out.extend_from_slice(s.name.as_bytes());
        out.push(0);
    }
    let strtab_len = out.len() - strtab;
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    // .symtab: the null symbol, then one per function.
    let symtab = out.len();
    out.extend_from_slice(&[0u8; SYM]);
    for (s, &name) in symbols.iter().zip(&names) {
        let mut e = [0u8; SYM];
        put(&mut e, 0, &name.to_le_bytes());
        e[4] = (STB_GLOBAL << 4) | STT_FUNC;
        put(&mut e, 6, &1u16.to_le_bytes()); // st_shndx: any defined section
        put(&mut e, 8, &CODE.wrapping_add(s.offset).to_le_bytes());
        put(&mut e, 16, &4u64.to_le_bytes());
        out.extend_from_slice(&e);
    }
    let symtab_len = out.len() - symtab;
    // .shstrtab
    let shstrtab = out.len();
    out.extend_from_slice(b"\0.symtab\0.strtab\0.shstrtab\0");
    let shstrtab_len = out.len() - shstrtab;
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let shoff = out.len();
    out.extend_from_slice(&[0u8; SHDR]);
    shdr(&mut out, 1, 2, symtab, symtab_len, 2, SYM as u64); // SHT_SYMTAB
    shdr(&mut out, 9, 3, strtab, strtab_len, 0, 0); // SHT_STRTAB
    shdr(&mut out, 17, 3, shstrtab, shstrtab_len, 0, 0);
    put(&mut out, 40, &(shoff as u64).to_le_bytes()); // e_shoff
    put(&mut out, 60, &4u16.to_le_bytes()); // e_shnum
    put(&mut out, 62, &3u16.to_le_bytes()); // e_shstrndx
    out
}
