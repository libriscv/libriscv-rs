//! Port of `Memory::lookup` and `Memory::print_backtrace` (memory.cpp):
//! symbolizing guest addresses with the program's ELF symbols.
//!
//! Unlike C++ built with `DEMANGLE_ENABLED`, names are not demangled
//! (that needs `__cxa_demangle` or a dependency).

use crate::elf;
use crate::error::Result;
use crate::machine::Machine;
use crate::registers::REG_RA;

/// The function an address is in (C++ `Memory::Callsite`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Callsite {
    pub name: String,
    pub address: u64,
    pub offset: u32,
    pub size: u64,
}

impl Default for Callsite {
    fn default() -> Self {
        Self {
            name: "(null)".into(),
            address: 0,
            offset: 0,
            size: 0,
        }
    }
}

impl Machine {
    /// Port of `Memory::lookup`: the function symbol at or below `address`
    /// with the highest address, or the default callsite (no name, all
    /// zero) when there is none.
    pub fn lookup(&self, address: u64) -> Result<Callsite> {
        let bin = self.memory.binary();
        // Backtraces can find null addresses.
        if !elf::validate(bin) || address == 0 {
            return Ok(Callsite::default());
        }
        // Add the load offset of dynamic programs.
        let address = self.memory.elf_base_address(address)?;
        let mut best: Option<(elf::Symbol, Option<String>)> = None;
        elf::for_each_symbol(bin, |sym, name| {
            if sym.symbol_type() == elf::STT_FUNC
                && address >= sym.st_value
                && best.as_ref().is_none_or(|(b, _)| sym.st_value > b.st_value)
            {
                best = Some((*sym, name.map(|n| String::from_utf8_lossy(n).into_owned())));
            }
            true
        })?;
        Ok(best.map_or_else(Callsite::default, |(sym, name)| Callsite {
            name: name.unwrap_or_else(|| "(invalid)".into()),
            address: sym.st_value,
            offset: address.wrapping_sub(sym.st_value) as u32,
            size: sym.st_size,
        }))
    }

    /// Port of `Memory::print_backtrace`: one line per frame, PC and (with
    /// `ra`) the return address, like `[0] 0x00000000000104B0 + 0x01c: main`.
    /// A frame without a symbol is an empty line.
    pub fn print_backtrace(&self, mut print: impl FnMut(&Self, &[u8]), ra: bool) -> Result<()> {
        let mut trace = |n: i32, addr: u64| -> Result<()> {
            let site = self.lookup(addr)?;
            if site.address == 0 && site.offset == 0 && site.size == 0 {
                print(self, &[]);
                return Ok(());
            }
            let prefix = if n >= 0 {
                format!("[{n}] ")
            } else {
                String::new()
            };
            let line = format!(
                "{prefix}0x{:016X} + 0x{:03x}: {}",
                site.address, site.offset, site.name
            );
            print(self, line.as_bytes());
            Ok(())
        };
        if ra {
            trace(0, self.cpu.pc())?;
            trace(1, self.cpu.reg(REG_RA))?;
        } else {
            trace(-1, self.cpu.pc())?;
        }
        Ok(())
    }
}
