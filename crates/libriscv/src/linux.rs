//! Port of `Machine::setup_linux`, `Machine::setup_argv` and
//! `Machine::stack_push` (`machine.cpp`, `machine_inline.hpp`) and
//! `util/auxvec.hpp`: the initial stack of a Linux process (argc, argv,
//! envp and the auxiliary vector) and of a newlib program.

use crate::elf;
use crate::error::{MachineError, Result};
use crate::machine::Machine;
use crate::memory::PAGE_SIZE;
use crate::registers::REG_SP;

/// Auxiliary vector entry types (`util/auxvec.hpp`).
pub mod auxv {
    pub const AT_NULL: u64 = 0;
    pub const AT_PHDR: u64 = 3;
    pub const AT_PHENT: u64 = 4;
    pub const AT_PHNUM: u64 = 5;
    pub const AT_PAGESZ: u64 = 6;
    pub const AT_BASE: u64 = 7;
    pub const AT_ENTRY: u64 = 9;
    pub const AT_UID: u64 = 11;
    pub const AT_EUID: u64 = 12;
    pub const AT_GID: u64 = 13;
    pub const AT_EGID: u64 = 14;
    pub const AT_PLATFORM: u64 = 15;
    pub const AT_HWCAP: u64 = 16;
    pub const AT_CLKTCK: u64 = 17;
    pub const AT_SECURE: u64 = 23;
    pub const AT_RANDOM: u64 = 25;
    pub const AT_HWCAP2: u64 = 26;
}

const PT_PHDR: u32 = 6;
const PHDR_SIZE: u64 = 56;

impl Machine {
    /// Port of `Machine::stack_push`: push `data` onto the guest stack
    /// (keeping word alignment) and return its address.
    pub fn stack_push(&mut self, data: &[u8]) -> Result<u64> {
        let sp = self.cpu.reg(REG_SP).wrapping_sub(data.len() as u64) & !7;
        self.cpu.set_reg(REG_SP, sp);
        self.copy_to_guest(sp, data)?;
        Ok(sp)
    }

    /// Push a string with its zero terminator (C++ `stack_push(std::string)`).
    pub fn stack_push_str(&mut self, s: &str) -> Result<u64> {
        let mut data = Vec::with_capacity(s.len() + 1);
        data.extend_from_slice(s.as_bytes());
        data.push(0);
        self.stack_push(&data)
    }

    /// Push a [`GuestPod`](crate::GuestPod) (C++ `stack_push(const T&)`).
    pub fn stack_push_value<T: crate::GuestPod>(&mut self, value: &T) -> Result<u64> {
        let mut data = Vec::with_capacity(T::SIZE);
        value.write_guest(&mut data);
        self.stack_push(&data)
    }

    /// Copy `words` below `sp`, 16-byte aligned, and make that the stack
    /// pointer and the initial stack.
    fn install_vector(&mut self, sp: u64, words: &[u64]) -> Result<()> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sp = sp.wrapping_sub(bytes.len() as u64) & !0xF;
        self.copy_to_guest(sp, &bytes)?;
        self.cpu.set_reg(REG_SP, sp);
        // Preserve argc/argv/envp (and the auxv) for the program lifetime.
        self.memory.set_stack_initial(sp);
        Ok(())
    }

    /// Port of `Machine::setup_argv`: put `argc`, `argv` and `envp` on the
    /// stack, for newlib and other minimal runtimes.
    pub fn setup_argv<A: AsRef<str>, E: AsRef<str>>(
        &mut self,
        args: &[A],
        env: &[E],
    ) -> Result<()> {
        let mut argv = vec![args.len() as u64];
        for a in args {
            argv.push(self.stack_push_str(a.as_ref())?);
        }
        argv.push(0);
        for e in env {
            argv.push(self.stack_push_str(e.as_ref())?);
        }
        argv.push(0);
        let sp = self.cpu.reg(REG_SP);
        self.install_vector(sp, &argv)
    }

    /// Port of `Machine::setup_linux`: build the initial stack of a Linux
    /// process below the stack pointer: 16 random bytes (`AT_RANDOM`), the
    /// platform string, the program headers (unless a `PT_PHDR` maps them),
    /// the argument and environment strings, then `argc`, `argv`, `envp`
    /// and the auxiliary vector.
    pub fn setup_linux<A: AsRef<str>, E: AsRef<str>>(
        &mut self,
        args: &[A],
        env: &[E],
    ) -> Result<()> {
        // Push below dst, keeping 8-byte alignment.
        fn push_down(m: &mut Machine, dst: &mut u64, data: &[u8]) -> Result<u64> {
            *dst = dst.wrapping_sub(data.len() as u64) & !7;
            m.copy_to_guest(*dst, data)?;
            Ok(*dst)
        }
        let mut dst = self.cpu.reg(REG_SP);

        let mut canary = [0u8; 16];
        crate::syscalls::host_random(&mut canary);
        let canary_addr = push_down(self, &mut dst, &canary)?;
        let platform_addr = push_down(self, &mut dst, b"RISC-V 64-bit\0")?;

        // Program headers
        let binary = std::sync::Arc::clone(self.memory.binary_arc());
        let ehdr = elf::header(&binary)?;
        let phnum = u64::from(ehdr.e_phnum);
        let phdrs_len = phnum * PHDR_SIZE;
        let phdrs = usize::try_from(ehdr.e_phoff)
            .ok()
            .zip(usize::try_from(ehdr.e_phoff.saturating_add(phdrs_len)).ok())
            .and_then(|(a, b)| binary.get(a..b))
            .ok_or(MachineError::invalid_program(
                "ELF program-headers are outside the binary",
                0,
            ))?;
        let mut phdr_location = 0;
        for i in 0..phnum {
            let ph = elf::program_header(&binary, ehdr.e_phoff + i * PHDR_SIZE)?;
            if ph.p_type == PT_PHDR {
                phdr_location = self.memory.elf_base_address(ph.p_vaddr)?;
                break;
            }
        }
        if phdr_location == 0 {
            for ph in phdrs.as_chunks::<{ PHDR_SIZE as usize }>().0.iter().rev() {
                push_down(self, &mut dst, ph)?;
            }
            phdr_location = dst;
        } else if self.memory.memcmp(phdrs, phdr_location)?.is_ne() {
            return Err(MachineError::invalid_program(
                "PT_PHDR program header is not loaded at the correct address",
                0,
            ));
        }

        // Arguments to main(), then the environment
        let mut push_str = |m: &mut Machine, s: &str| -> Result<u64> {
            let mut data = s.as_bytes().to_vec();
            data.push(0);
            push_down(m, &mut dst, &data)
        };
        let mut argv = vec![args.len() as u64];
        for a in args {
            argv.push(push_str(self, a.as_ref())?);
        }
        argv.push(0);
        for e in env {
            argv.push(push_str(self, e.as_ref())?);
        }
        argv.push(0);

        // Auxiliary vector
        use auxv::*;
        let start = self.memory.start_address();
        argv.extend_from_slice(&[
            AT_PAGESZ,
            PAGE_SIZE,
            AT_CLKTCK,
            100,
            AT_PHDR,
            phdr_location,
            AT_PHENT,
            PHDR_SIZE,
            AT_PHNUM,
            phnum,
            AT_BASE,
            start & !0xFF_FFFF,
            AT_ENTRY,
            start,
            AT_HWCAP,
            0,
            AT_HWCAP2,
            0,
            AT_UID,
            1000,
            AT_EUID,
            0,
            AT_GID,
            0,
            AT_EGID,
            0,
            AT_SECURE,
            0,
            AT_PLATFORM,
            platform_addr,
            AT_RANDOM,
            canary_addr,
            AT_NULL,
            0,
        ]);
        // From here on the stack starts, pointing at argc.
        self.install_vector(dst, &argv)
    }
}
