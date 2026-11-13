//! Port of `elf.hpp` and the section/symbol parts of `memory_elf.cpp`.
//!
//! ELF64 little-endian only. C++ reinterprets the binary as packed structs;
//! here every field is read with a bounds-checked little-endian load, so a
//! malformed binary produces `INVALID_PROGRAM` instead of an out-of-bounds read.

use crate::error::{MachineError, Result};

pub(crate) const ELFCLASS64: u8 = 2;
pub(crate) const ET_EXEC: u16 = 2;
pub(crate) const ET_DYN: u16 = 3;
pub(crate) const EM_RISCV: u16 = 243;

pub(crate) const PT_LOAD: u32 = 1;
pub(crate) const PT_GNU_STACK: u32 = 0x6474_e551;
pub(crate) const PT_GNU_RELRO: u32 = 0x6474_e552;

pub(crate) const PF_X: u32 = 1 << 0;
pub(crate) const PF_W: u32 = 1 << 1;
pub(crate) const PF_R: u32 = 1 << 2;

pub(crate) const SHT_NOBITS: u32 = 8;
pub(crate) const SHF_ALLOC: u64 = 0x2;
pub(crate) const SHF_EXECINSTR: u64 = 0x4;

/// `sizeof(Elf64_Ehdr)`.
pub(crate) const HEADER_SIZE: usize = 64;
/// `sizeof(Elf64_Phdr)`.
pub(crate) const PHDR_SIZE: usize = 56;
/// `sizeof(Elf64_Shdr)`.
pub(crate) const SHDR_SIZE: usize = 64;
/// `sizeof(Elf64_Sym)`.
pub(crate) const SYM_SIZE: usize = 24;
/// `sizeof(Elf64_Rela)`.
pub(crate) const RELA_SIZE: usize = 24;

const fn invalid_offset(ofs: u64) -> MachineError {
    MachineError::invalid_program("Invalid ELF offset", ofs)
}

/// Bounds-checked little-endian field reads from the binary.
fn bytes<const N: usize>(bin: &[u8], ofs: u64) -> Result<[u8; N]> {
    usize::try_from(ofs)
        .ok()
        .and_then(|o| bin.get(o..))
        .and_then(|s| s.first_chunk::<N>())
        .copied()
        .ok_or(invalid_offset(ofs))
}
pub(crate) fn u16_at(bin: &[u8], ofs: u64) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes(bin, ofs)?))
}
pub(crate) fn u32_at(bin: &[u8], ofs: u64) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes(bin, ofs)?))
}
pub(crate) fn u64_at(bin: &[u8], ofs: u64) -> Result<u64> {
    Ok(u64::from_le_bytes(bytes(bin, ofs)?))
}

/// `Elf64_Ehdr` (the fields the loader uses).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Header {
    pub e_type: u16,
    pub e_machine: u16,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_phnum: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

/// `Elf64_Phdr`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProgramHeader {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
}

/// `Elf64_Shdr`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SectionHeader {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
}

/// `Elf64_Sym`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Symbol {
    pub st_name: u32,
    pub st_info: u8,
    pub st_value: u64,
    pub st_size: u64,
}

/// `STT_FUNC`.
pub(crate) const STT_FUNC: u8 = 2;

impl Symbol {
    /// `ELF64_ST_TYPE` (C++ `Elf::SymbolType`).
    pub(crate) const fn symbol_type(&self) -> u8 {
        self.st_info & 0xF
    }
}

/// `Elf64_Rela`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rela {
    pub r_offset: u64,
    pub r_info: u64,
}

impl Rela {
    pub(crate) const fn sym(&self) -> u64 {
        self.r_info >> 32
    }
    pub(crate) const fn rtype(&self) -> u32 {
        self.r_info as u32
    }
}

/// C++ `Elf<8>::validate`: magic and ELFCLASS64.
pub(crate) fn validate(bin: &[u8]) -> bool {
    bin.len() >= HEADER_SIZE && bin.get(..4) == Some(b"\x7fELF") && bin.get(4) == Some(&ELFCLASS64)
}

pub(crate) fn header(bin: &[u8]) -> Result<Header> {
    Ok(Header {
        e_type: u16_at(bin, 16)?,
        e_machine: u16_at(bin, 18)?,
        e_entry: u64_at(bin, 24)?,
        e_phoff: u64_at(bin, 32)?,
        e_shoff: u64_at(bin, 40)?,
        e_flags: u32_at(bin, 48)?,
        e_phnum: u16_at(bin, 56)?,
        e_shnum: u16_at(bin, 60)?,
        e_shstrndx: u16_at(bin, 62)?,
    })
}

pub(crate) fn program_header(bin: &[u8], ofs: u64) -> Result<ProgramHeader> {
    Ok(ProgramHeader {
        p_type: u32_at(bin, ofs)?,
        p_flags: u32_at(bin, ofs.wrapping_add(4))?,
        p_offset: u64_at(bin, ofs.wrapping_add(8))?,
        p_vaddr: u64_at(bin, ofs.wrapping_add(16))?,
        p_filesz: u64_at(bin, ofs.wrapping_add(32))?,
        p_memsz: u64_at(bin, ofs.wrapping_add(40))?,
    })
}

pub(crate) fn section_header(bin: &[u8], ofs: u64) -> Result<SectionHeader> {
    Ok(SectionHeader {
        sh_name: u32_at(bin, ofs)?,
        sh_type: u32_at(bin, ofs.wrapping_add(4))?,
        sh_flags: u64_at(bin, ofs.wrapping_add(8))?,
        sh_addr: u64_at(bin, ofs.wrapping_add(16))?,
        sh_offset: u64_at(bin, ofs.wrapping_add(24))?,
        sh_size: u64_at(bin, ofs.wrapping_add(32))?,
    })
}

fn symbol(bin: &[u8], ofs: u64) -> Result<Symbol> {
    Ok(Symbol {
        st_name: u32_at(bin, ofs)?,
        st_info: u8::from_le_bytes(bytes(bin, ofs.wrapping_add(4))?),
        st_value: u64_at(bin, ofs.wrapping_add(8))?,
        st_size: u64_at(bin, ofs.wrapping_add(16))?,
    })
}

pub(crate) fn rela(bin: &[u8], ofs: u64) -> Result<Rela> {
    Ok(Rela {
        r_offset: u64_at(bin, ofs)?,
        r_info: u64_at(bin, ofs.wrapping_add(8))?,
    })
}

/// The NUL-terminated string at `ofs`, bounded by `end`.
fn cstr_at(bin: &[u8], ofs: u64, end: usize) -> Option<&[u8]> {
    let start = usize::try_from(ofs).ok()?;
    let s = bin.get(start..end.min(bin.len()))?;
    let len = s.iter().position(|&b| b == 0)?;
    s.get(..len)
}

/// Port of `Memory::section_by_name`. Throws on a malformed section table
/// (as C++ does), returns `None` when no section has that name.
pub(crate) fn section_by_name(bin: &[u8], name: &str) -> Result<Option<SectionHeader>> {
    let elf = header(bin)?;
    let sh_end_offset = elf
        .e_shoff
        .wrapping_add(u64::from(elf.e_shnum) * SHDR_SIZE as u64);
    let size = bin.len() as u64;
    if elf.e_shoff > size {
        return Err(MachineError::invalid_program(
            "Invalid section header offset",
            elf.e_shoff,
        ));
    }
    if sh_end_offset < elf.e_shoff || sh_end_offset > size {
        return Err(MachineError::invalid_program(
            "Invalid section header offset",
            sh_end_offset,
        ));
    }
    if elf.e_shnum == 0 || elf.e_shnum > 64 {
        return Err(MachineError::invalid_program(
            "Invalid section header count",
            u64::from(elf.e_shnum),
        ));
    }
    if elf.e_shstrndx >= elf.e_shnum {
        return Err(MachineError::invalid_program(
            "Invalid section header strtab index",
            0,
        ));
    }
    let shdr_at = |i: u16| section_header(bin, elf.e_shoff + u64::from(i) * SHDR_SIZE as u64);
    let shstrtab = shdr_at(elf.e_shstrndx)?;
    // C++ elf_offset<char>(sh_offset) checks that one byte is inside.
    if shstrtab.sh_offset.wrapping_add(1) > size || shstrtab.sh_offset.wrapping_add(1) == 0 {
        return Err(invalid_offset(shstrtab.sh_offset));
    }
    for i in 0..elf.e_shnum {
        let sh = shdr_at(i)?;
        let name_offset = shstrtab.sh_offset.wrapping_add(u64::from(sh.sh_name));
        if name_offset < shstrtab.sh_offset || name_offset >= size {
            return Err(MachineError::invalid_program(
                "Invalid ELF string offset",
                0,
            ));
        }
        // strnlen to the end of the binary: an unterminated name never matches.
        if let Some(shname) = cstr_at(bin, name_offset, bin.len()) {
            if shname == name.as_bytes() {
                return Ok(Some(sh));
            }
        } else if bin.get(name_offset as usize..) == Some(name.as_bytes()) {
            return Ok(Some(sh));
        }
    }
    Ok(None)
}

/// Port of `Memory::section_by_name_validated`: also rejects `SHT_NOBITS`
/// sections and file ranges outside the binary.
pub(crate) fn section_by_name_validated(bin: &[u8], name: &str) -> Result<Option<SectionHeader>> {
    let Some(sh) = section_by_name(bin, name)? else {
        return Ok(None);
    };
    if sh.sh_type == SHT_NOBITS {
        return Ok(None);
    }
    let file_end = sh.sh_offset.wrapping_add(sh.sh_size);
    if file_end < sh.sh_offset || file_end > bin.len() as u64 {
        return Ok(None);
    }
    Ok(Some(sh))
}

/// Port of `Memory::elf_symbol_name`.
fn symbol_name<'a>(bin: &'a [u8], strtab: &SectionHeader, st_name: u32) -> Option<&'a [u8]> {
    if u64::from(st_name) >= strtab.sh_size {
        return None;
    }
    // section_by_name_validated() guarantees the strtab is inside the binary.
    let end = usize::try_from(strtab.sh_offset + strtab.sh_size).ok()?;
    cstr_at(bin, strtab.sh_offset + u64::from(st_name), end)
}

/// Port of `Memory::for_each_symbol`. The name is `None` when it is invalid.
pub(crate) fn for_each_symbol(
    bin: &[u8],
    mut f: impl FnMut(&Symbol, Option<&[u8]>) -> bool,
) -> Result<()> {
    if bin.is_empty() {
        return Ok(());
    }
    let Some(sym_hdr) = section_by_name_validated(bin, ".symtab")? else {
        return Ok(());
    };
    let Some(str_hdr) = section_by_name_validated(bin, ".strtab")? else {
        return Ok(());
    };
    let count = sym_hdr.sh_size / SYM_SIZE as u64;
    for i in 0..count {
        let sym = symbol(bin, sym_hdr.sh_offset + i * SYM_SIZE as u64)?;
        if !f(&sym, symbol_name(bin, &str_hdr, sym.st_name)) {
            break;
        }
    }
    Ok(())
}

/// Port of `Memory::resolve_symbol`: the first symbol with this name.
pub(crate) fn resolve_symbol(bin: &[u8], name: &str) -> Result<Option<Symbol>> {
    let mut result = None;
    for_each_symbol(bin, |sym, symname| {
        if symname == Some(name.as_bytes()) {
            result = Some(*sym);
            return false;
        }
        true
    })?;
    Ok(result)
}

/// Port of `Memory::elf_sym_index`.
pub(crate) fn symbol_at(bin: &[u8], symtab: &SectionHeader, idx: u64) -> Result<Symbol> {
    if idx >= symtab.sh_size / SYM_SIZE as u64 {
        return Err(MachineError::invalid_program(
            "ELF Symtab section index overflow",
            0,
        ));
    }
    symbol(bin, symtab.sh_offset + idx * SYM_SIZE as u64)
}
/// The program interpreter (`PT_INTERP`) of an executable, if it has one:
/// dynamically linked executables are run by their interpreter (C++
/// `Elf::is_dynamic`, which also returns the interpreter). `None` for
/// static (and static-pie) executables and for anything unparsable.
pub fn interpreter(bin: &[u8]) -> Option<String> {
    const PT_INTERP: u32 = 3;
    let hdr = header(bin).ok()?;
    (0..u64::from(hdr.e_phnum)).find_map(|i| {
        let ph = program_header(bin, hdr.e_phoff.checked_add(i * 56)?).ok()?;
        if ph.p_type != PT_INTERP {
            return None;
        }
        let start = usize::try_from(ph.p_offset).ok()?;
        let end = start.checked_add(usize::try_from(ph.p_filesz).ok()?)?;
        let raw = bin.get(start..end)?;
        let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        Some(String::from_utf8_lossy(raw.get(..len)?).into_owned())
    })
}
