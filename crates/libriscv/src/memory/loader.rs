//! Port of the ELF loading parts of `memory.cpp` (`binary_loader`,
//! `binary_load_ph`, `serialize_execute_segment`) and `memory_elf.cpp`
//! (`elf_base_address`, relocations), for the flat arena without paging.
//!
//! Not ported: shared read-only images (`prepare_shared_rodata`,
//! `publish_shared_rodata`) and `default_exit_function`.

use std::sync::Arc;

use super::{BRK_MAX, DYLINK_BASE, Memory, PAGE_SIZE, RWREAD_BEGIN, SIGRETURN_OFFSET};
use crate::decoder_cache::{COMPRESSED, DecodedSegment};
use crate::elf::{self, ProgramHeader};
use crate::error::{MachineError, Result};
use crate::options::MachineOptions;

const ELFHDR_FLAGS_RVC: u32 = 0x1;
const ELFHDR_FLAGS_RVE: u32 = 0x8;

/// The exit function placed in its own execute segment when the program
/// doesn't provide one: STOP, then a jump back to it, followed by the signal
/// return trampoline at `SIGRETURN_OFFSET`.
const EXIT_CODE: [u8; 20] = [
    0x73, 0x00, 0xf0, 0x7f, // STOP: 0x7ff00073
    0x6f, 0xf0, 0xdf, 0xff, // JMP -4: 0xffdff06f
    0x93, 0x08, 0xb0, 0x08, // LI a7, 139 (rt_sigreturn)
    0x73, 0x00, 0x00, 0x00, // ECALL
    0x6f, 0xf0, 0xdf, 0xff, // JMP -4: 0xffdff06f
];

const fn invalid(message: &'static str) -> MachineError {
    MachineError::invalid_program(message, 0)
}

impl Memory {
    /// Port of `Memory::elf_base_address`.
    pub(crate) fn elf_base_address(&self, offset: u64) -> Result<u64> {
        if self.is_dynamic {
            DYLINK_BASE
                .checked_add(offset)
                .ok_or(invalid("Bogus virtual address + offset"))
        } else {
            Ok(offset)
        }
    }

    /// Port of `Memory::binary_loader`. Returns the first execute segment
    /// of the program.
    pub(super) fn binary_loader(
        &mut self,
        options: &MachineOptions,
    ) -> Result<Option<Arc<DecodedSegment>>> {
        let binary = Arc::clone(&self.binary);
        let bin: &[u8] = &binary;
        if bin.len() < elf::HEADER_SIZE {
            return Err(invalid("ELF program too short"));
        }
        if !elf::validate(bin) {
            return Err(invalid(
                "Invalid ELF header! Expected a 64-bit RISC-V ELF binary",
            ));
        }
        let hdr = elf::header(bin)?;
        let is_static = hdr.e_type == elf::ET_EXEC;
        self.is_dynamic = hdr.e_type == elf::ET_DYN;
        if !is_static && !self.is_dynamic {
            return Err(invalid(
                "ELF program is not an executable type. Trying to load an object file?",
            ));
        }
        if hdr.e_machine != elf::EM_RISCV {
            return Err(invalid(
                "ELF program is not a RISC-V executable. Wrong architecture.",
            ));
        }
        if hdr.e_flags & ELFHDR_FLAGS_RVC != 0 && !COMPRESSED {
            return Err(invalid(
                "ELF is a RISC-V RVC executable, however C-extension is not enabled.",
            ));
        }
        if hdr.e_flags & ELFHDR_FLAGS_RVE != 0 {
            return Err(invalid(
                "ELF is a RISC-V RVE executable, however E-extension is not supported.",
            ));
        }

        let program_headers = u64::from(hdr.e_phnum);
        if program_headers == 0 {
            return Err(invalid("ELF with no program-headers"));
        }
        if program_headers >= 16 {
            return Err(invalid("ELF with too many program-headers"));
        }
        if hdr.e_phoff > 0x4000 {
            return Err(invalid("ELF program-headers have bogus offset"));
        }
        if hdr.e_phoff + program_headers * elf::PHDR_SIZE as u64 > bin.len() as u64 {
            return Err(invalid("ELF program-headers are outside the binary"));
        }
        if hdr.e_phoff % 8 != 0 {
            return Err(invalid("ELF program-headers are not aligned"));
        }
        let phdrs = (0..program_headers)
            .map(|i| elf::program_header(bin, hdr.e_phoff + i * elf::PHDR_SIZE as u64))
            .collect::<Result<Vec<ProgramHeader>>>()?;

        self.start_address = self.elf_base_address(hdr.e_entry)?;
        self.heap_address = 0;

        let mut execute_segments = Vec::new();
        for (k, ph) in phdrs.iter().enumerate() {
            let vaddr = self.elf_base_address(ph.p_vaddr)?;

            // Detect overlapping segments. No normal ELF has them.
            for prev in phdrs.iter().take(k) {
                let prev_vaddr = self.elf_base_address(prev.p_vaddr)?;
                if ph.p_type == elf::PT_LOAD
                    && prev.p_type == elf::PT_LOAD
                    && prev_vaddr < vaddr.wrapping_add(ph.p_filesz)
                    && prev_vaddr.wrapping_add(prev.p_filesz) > vaddr
                {
                    return Err(invalid("Overlapping ELF segments"));
                }
            }

            match ph.p_type {
                elf::PT_LOAD if options.load_program => {
                    self.binary_load_ph(options, ph, vaddr)?;
                    if ph.p_flags & elf::PF_X != 0 {
                        execute_segments.push(*ph);
                    }
                }
                // Executable stack marker: ignored. RELRO: not enforced (C++
                // has it commented out too).
                elf::PT_GNU_STACK | elf::PT_GNU_RELRO => {}
                _ => {}
            }

            let endm =
                vaddr.wrapping_add(ph.p_memsz).wrapping_add(PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            if self.heap_address < endm {
                self.heap_address = endm;
            }
        }

        // The base mmap address starts at heap start + BRK_MAX.
        self.mmap_address = self.heap_address.wrapping_add(BRK_MAX);
        self.brk_address = self.heap_address;

        // Default stack
        let stack_size = u64::from(options.stack_size);
        self.stack_address = self.mmap_allocate(stack_size).wrapping_add(stack_size);

        // The arena boundaries are final once every PT_LOAD is in place.
        if self.arena_size >= self.initial_rodata_end {
            self.read_boundary = self
                .arena_size
                .min(self.arena_size.wrapping_sub(RWREAD_BEGIN));
            self.write_boundary = self
                .arena_size
                .min(self.arena_size.wrapping_sub(self.initial_rodata_end));
        } else {
            self.initial_rodata_end = 0;
            self.read_boundary = 0;
            self.write_boundary = 0;
        }

        // Install our own exit function as a separate execute segment.
        if self.exit_address == 0 {
            let host_page = self.mmap_allocate(PAGE_SIZE);
            self.exit_address = host_page;
            self.sigreturn_address = host_page.wrapping_add(SIGRETURN_OFFSET);
            if host_page.wrapping_add(EXIT_CODE.len() as u64) <= self.arena_size {
                // Fits, as checked above.
                if let Some(dst) = usize::try_from(host_page)
                    .ok()
                    .and_then(|a| self.arena.get_mut(a..a + EXIT_CODE.len()))
                {
                    dst.copy_from_slice(&EXIT_CODE);
                }
            }
            self.create_execute_segment(&EXIT_CODE, host_page, false)?;
        }

        // Now that the program's boundaries are known, generate the execute
        // segments.
        let mut first_segment = None;
        if options.load_program {
            for ph in &execute_segments {
                let vaddr = self.elf_base_address(ph.p_vaddr)?;
                let segment = self.serialize_execute_segment(options, ph, vaddr)?;
                first_segment.get_or_insert(segment);
            }
            if self.is_dynamic {
                self.dynamic_linking()?;
            }
        }

        if options.verbose_loader {
            println!("* Entry is at {:#x}", self.start_address);
        }
        Ok(first_segment)
    }

    /// Port of `Memory::binary_load_ph` (flat arena, no paging).
    fn binary_load_ph(
        &mut self,
        options: &MachineOptions,
        ph: &ProgramHeader,
        vaddr: u64,
    ) -> Result<()> {
        let len = ph.p_filesz;
        let size = self.binary.len() as u64;
        if size <= ph.p_offset || ph.p_offset.wrapping_add(len) < ph.p_offset {
            return Err(invalid("Bogus ELF program segment offset"));
        }
        if size < ph.p_offset + len {
            return Err(invalid("Not enough room for ELF program segment"));
        }
        if vaddr.wrapping_add(len) < vaddr {
            return Err(invalid("Bogus ELF segment virtual base"));
        }
        if options.verbose_loader {
            println!(
                "* Loading program of size {len} from offset {:#x} to virtual {vaddr:#x} -> {:#x}",
                ph.p_offset,
                vaddr + len
            );
        }
        // Nothing further to do for an empty segment.
        if len == 0 {
            return Ok(());
        }
        let read = ph.p_flags & elf::PF_R != 0;
        let write = ph.p_flags & elf::PF_W != 0;
        let exec = ph.p_flags & elf::PF_X != 0;
        if options.verbose_loader {
            println!(
                "* Program segment readable: {} writable: {}  executable: {}",
                u8::from(read),
                u8::from(write),
                u8::from(exec)
            );
        }
        // Nothing more to do here, if execute-only.
        if exec && !read {
            return Ok(());
        }
        if exec && write && !options.allow_write_exec_segment {
            return Err(invalid(
                "Insecure ELF has writable executable code (Disable check in MachineOptions)",
            ));
        }
        if exec && (read || write) && options.enforce_exec_only {
            return Err(invalid("Execute segment must be execute-only"));
        }

        // Write directly to the arena (the rodata boundary is set afterwards).
        if vaddr + len > self.arena_size {
            return Err(invalid("ELF segment exceeds arena size"));
        }
        let src = usize::try_from(ph.p_offset)
            .ok()
            .and_then(|o| {
                self.binary
                    .get(o..o.checked_add(usize::try_from(len).ok()?)?)
            })
            .ok_or(invalid("Not enough room for ELF program segment"))?;
        let dst = usize::try_from(vaddr)
            .ok()
            .and_then(|a| self.arena.get_mut(a..a + src.len()))
            .ok_or(invalid("ELF segment exceeds arena size"))?;
        dst.copy_from_slice(src);

        // Track the read-only boundary.
        if read && !write {
            self.initial_rodata_end = self.initial_rodata_end.max(vaddr + len);
        }
        // set_page_attr() is a no-op without paging.
        Ok(())
    }

    /// Port of `Memory::serialize_execute_segment`.
    fn serialize_execute_segment(
        &mut self,
        options: &MachineOptions,
        ph: &ProgramHeader,
        mut vaddr: u64,
    ) -> Result<Arc<DecodedSegment>> {
        let binary = Arc::clone(&self.binary);
        let bin: &[u8] = &binary;
        let mut exlen = ph.p_filesz;
        let mut data_offset = ph.p_offset;

        // Zig's ELF writer is insane, so there is an option to disable the
        // .text section segment reduction.
        if !options.ignore_text_section {
            // Section and segment addresses are compared as written in the
            // ELF, before any dynamic base is applied.
            let seg_begin = ph.p_vaddr;
            let seg_end = seg_begin.wrapping_add(ph.p_filesz);
            let texthdr = elf::section_by_name_validated(bin, ".text")?;
            if let Some(text) = texthdr {
                let text_end = text.sh_addr.wrapping_add(text.sh_size);
                if seg_end >= seg_begin
                    && text.sh_addr >= seg_begin
                    && text.sh_size <= exlen
                    && text_end >= text.sh_addr
                    && text_end <= seg_end
                {
                    // Widen the range to cover every other executable section
                    // in this segment.
                    let mut lo = text.sh_addr;
                    let mut hi = text_end;
                    let hdr = elf::header(bin)?;
                    for i in 0..u64::from(hdr.e_shnum) {
                        let sh = elf::section_header(bin, hdr.e_shoff + i * elf::SHDR_SIZE as u64)?;
                        const ALLOC_EXEC: u64 = elf::SHF_ALLOC | elf::SHF_EXECINSTR;
                        if sh.sh_flags & ALLOC_EXEC != ALLOC_EXEC || sh.sh_type == elf::SHT_NOBITS {
                            continue;
                        }
                        let sh_end = sh.sh_addr.wrapping_add(sh.sh_size);
                        if sh.sh_addr < seg_begin || sh_end < sh.sh_addr || sh_end > seg_end {
                            continue;
                        }
                        lo = lo.min(sh.sh_addr);
                        hi = hi.max(sh_end);
                    }
                    // The bytes come from the segment, not from sh_offset.
                    data_offset = ph.p_offset + (lo - seg_begin);
                    vaddr = self.elf_base_address(lo)?;
                    exlen = hi - lo;
                }
            }
        }

        // binary_load_ph() has verified that the segment is inside the binary.
        let code = usize::try_from(data_offset)
            .ok()
            .and_then(|o| bin.get(o..o.checked_add(usize::try_from(exlen).ok()?)?))
            .ok_or(invalid("Not enough room for ELF program segment"))?;
        // The segment is execute-only when neither R nor W is set.
        let execute_only = ph.p_flags & (elf::PF_R | elf::PF_W) == 0;
        self.create_execute_segment(code, vaddr, execute_only)
    }

    /// Port of `Memory::dynamic_linking`.
    fn dynamic_linking(&mut self) -> Result<()> {
        self.relocate_section(".rela.dyn", ".dynsym")?;
        self.relocate_section(".rela.plt", ".symtab")
    }

    /// Port of `Memory::relocate_section`.
    fn relocate_section(&mut self, section_name: &str, sym_section: &str) -> Result<()> {
        const R_RISCV_64: u32 = 0x2;
        const R_RISCV_RELATIVE: u32 = 0x3;
        const R_RISCV_JUMPSLOT: u32 = 0x5;
        let binary = Arc::clone(&self.binary);
        let bin: &[u8] = &binary;
        let Some(rela) = elf::section_by_name_validated(bin, section_name)? else {
            return Ok(());
        };
        let Some(dyn_hdr) = elf::section_by_name_validated(bin, sym_section)? else {
            return Ok(());
        };
        let count = rela.sh_size / elf::RELA_SIZE as u64;
        for i in 0..count {
            let r = elf::rela(bin, rela.sh_offset + i * elf::RELA_SIZE as u64)?;
            let sym = elf::symbol_at(bin, &dyn_hdr, r.sym())?;
            match r.rtype() {
                0 | R_RISCV_JUMPSLOT => {}
                R_RISCV_64 => {
                    let addr = self.elf_base_address(r.r_offset)?;
                    let value = self.elf_base_address(sym.st_value)?;
                    self.write_u64(addr, value)?;
                }
                R_RISCV_RELATIVE => {
                    // C++ writes the symbol value, not base + addend.
                    let addr = self.elf_base_address(r.r_offset)?;
                    self.write_u64(addr, sym.st_value)?;
                }
                rtype => {
                    return Err(MachineError::invalid_program(
                        "Unknown relocation type",
                        u64::from(rtype),
                    ));
                }
            }
        }
        Ok(())
    }
}
