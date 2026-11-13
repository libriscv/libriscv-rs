//! Saving and reusing the Cranelift JIT's code across runs: the JIT
//! counterpart of C++'s translation cache (`.so` files `dlopen`ed by hash)
//! and embedded translations (self-registering code linked into the host),
//! in `tr_translate.cpp`. Unlike C++, nothing is translated to C: the
//! saved code is exactly what the JIT compiled, so it behaves exactly like
//! the JIT.
//!
//! A machine with a [`JitCacheOutput`] records the [`Image`] of every
//! segment it compiles. [`JitCacheOutput::write_object`] writes them as one
//! ELF relocatable object, to link into a host program (embedding), and
//! [`JitCacheOutput::write_shared_object`] links that into a shared object
//! with the system C compiler driver (`$CC`, else `cc`).
//!
//! Code linked into the program is found automatically, and shared objects
//! are added with [`load_shared_object`], both into one process-wide
//! registry. When a machine creates an execute segment, it first looks for
//! code for it there (`MachineOptions::jit_cache`); only otherwise does the
//! JIT compile it. Code is keyed on the segment (address, length, contents,
//! execute-only, libc fast path), the memory layout baked into the code
//! and a fingerprint of the build ([`build_fingerprint`](super::emit::build_fingerprint)),
//! so a file from another build, host CPU or memory layout is ignored, not
//! run.
//!
//! # Object format
//!
//! See `libriscv_jit_rt`'s `linked` module for the records and the table
//! that lead to each segment's code and metadata. The code is the image
//! as the JIT links it, with its libcalls as `R_X86_64_GOTPCREL`
//! relocations against the C library's functions. The metadata is
//! little-endian `u64`s: [`META_MAGIC`], the key (see [`CacheKey::write`]),
//! the trampoline offset, the entry count and the entries (`u32`, see
//! [`Image::entries`]), then the patch count and the patched decoder
//! indices (`u32`).

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cranelift_codegen::ir::LibCall;
use libriscv_jit_rt::{
    CachedUnit, CodeMemory, LinkedCode, MemoryLayout, RECORD_MAGIC, RECORD_WORDS, SECTION,
    TABLE_MAGIC, TABLE_SYMBOL,
};

use super::emit::{Compiled, Image, build_fingerprint, libcall_symbol};

/// First word of a unit's metadata ("RVJITM01").
const META_MAGIC: u64 = u64::from_le_bytes(*b"RVJITM01");

pub(crate) use crate::codegen::hash64;

/// What cached code is looked up by: everything the code compiled for a
/// segment depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    /// [`build_fingerprint`].
    pub build: u64,
    pub vaddr: u64,
    pub len: u64,
    pub crc32c: u32,
    /// [`hash64`] of the code.
    pub hash: u64,
    pub execute_only: bool,
    /// The libc fast path patched the segment.
    pub libc_fastpath: bool,
    pub layout: MemoryLayout,
}

impl CacheKey {
    /// The key of the segment `code` at `vaddr` (whose CRC32-C is
    /// `crc32c`), or `None` where the JIT isn't supported.
    pub(crate) fn new(
        code: &[u8],
        vaddr: u64,
        crc32c: u32,
        execute_only: bool,
        libc_fastpath: bool,
        layout: MemoryLayout,
    ) -> Option<Self> {
        Some(Self {
            build: build_fingerprint()?,
            vaddr,
            len: code.len() as u64,
            crc32c,
            hash: hash64(code),
            execute_only,
            libc_fastpath,
            layout,
        })
    }

    const WORDS: usize = 11;

    fn write(&self, out: &mut Vec<u8>) {
        let flags = u64::from(self.execute_only) | u64::from(self.libc_fastpath) << 1;
        let l = &self.layout;
        for w in [
            self.build,
            self.vaddr,
            self.len,
            u64::from(self.crc32c),
            self.hash,
            flags,
            l.mask,
            l.read.0,
            l.read.1,
            l.write.0,
            l.write.1,
        ] {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    fn read(r: &mut Reader<'_>) -> Option<Self> {
        let mut w = [0u64; Self::WORDS];
        for x in &mut w {
            *x = r.u64()?;
        }
        let [build, vaddr, len, crc, hash, flags, mask, rb, rl, wb, wl] = w;
        Some(Self {
            build,
            vaddr,
            len,
            crc32c: u32::try_from(crc).ok()?,
            hash,
            execute_only: flags & 1 != 0,
            libc_fastpath: flags & 2 != 0,
            layout: MemoryLayout {
                mask,
                read: (rb, rl),
                write: (wb, wl),
            },
        })
    }
}

/// Reads metadata, refusing anything truncated.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let (head, tail) = self.0.split_at_checked(n)?;
        self.0 = tail;
        Some(head)
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    /// A count of `u32`s that must fit in the rest.
    fn count(&mut self) -> Option<usize> {
        let n = usize::try_from(self.u64()?).ok()?;
        (n <= self.0.len() / 4).then_some(n)
    }
}

/// The metadata of `image` for `key`.
fn metadata(key: &CacheKey, image: &Image) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&META_MAGIC.to_le_bytes());
    key.write(&mut out);
    out.extend_from_slice(&(image.trampoline as u64).to_le_bytes());
    out.extend_from_slice(&(image.entries.len() as u64).to_le_bytes());
    for &e in &image.entries {
        out.extend_from_slice(&e.to_le_bytes());
    }
    out.extend_from_slice(&(image.patch.len() as u64).to_le_bytes());
    for &p in &image.patch {
        out.extend_from_slice(&u32::try_from(p).ok()?.to_le_bytes());
    }
    Some(out)
}

/// A unit's metadata, besides its key.
#[derive(Debug, PartialEq, Eq)]
struct Meta {
    trampoline: usize,
    entries: Vec<u32>,
    patch: Vec<usize>,
}

/// A segment's cached code, ready to be entered.
#[derive(Debug)]
struct Entry {
    meta: Meta,
    code: LinkedCode,
}

/// Parse a unit's metadata.
fn parse(meta: &[u8]) -> Option<(CacheKey, Meta)> {
    let mut r = Reader(meta);
    if r.u64()? != META_MAGIC {
        return None;
    }
    let key = CacheKey::read(&mut r)?;
    let trampoline = usize::try_from(r.u64()?).ok()?;
    let n = r.count()?;
    let entries = (0..n).map(|_| r.u32()).collect::<Option<Vec<_>>>()?;
    let n = r.count()?;
    let patch = (0..n)
        .map(|_| r.u32().map(|p| p as usize))
        .collect::<Option<Vec<_>>>()?;
    // Every patched entry must have a function.
    if patch
        .iter()
        .any(|&p| entries.get(p).is_none_or(|&e| e == CodeMemory::NO_ENTRY))
    {
        return None;
    }
    Some((
        key,
        Meta {
            trampoline,
            entries,
            patch,
        },
    ))
}

/// The process-wide registry of cached code.
#[derive(Debug, Default)]
struct Registry {
    /// The program's embedded code has been added.
    embedded: bool,
    units: HashMap<CacheKey, Entry>,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);
/// The registry has units (checked without locking, before hashing a
/// segment for nothing).
static HAS_UNITS: AtomicBool = AtomicBool::new(false);

fn registry() -> MutexGuard<'static, Option<Registry>> {
    let mut guard = REGISTRY.lock().unwrap_or_else(PoisonError::into_inner);
    let registry = guard.get_or_insert_with(Registry::default);
    if !registry.embedded {
        registry.embedded = true;
        add(registry, libriscv_jit_rt::embedded_units());
    }
    guard
}

/// Add `units` to the registry, keeping existing code for a key (as C++
/// does). Returns the number of units added.
fn add(registry: &mut Registry, units: Vec<CachedUnit>) -> usize {
    let mut added = 0;
    for unit in units {
        let Some((key, meta)) = parse(&unit.meta) else {
            continue;
        };
        if let std::collections::hash_map::Entry::Vacant(v) = registry.units.entry(key) {
            v.insert(Entry {
                meta,
                code: unit.code,
            });
            added += 1;
        }
    }
    if !registry.units.is_empty() {
        HAS_UNITS.store(true, Ordering::Release);
    }
    added
}

/// True if there is any cached code (embedded in the program or from
/// loaded shared objects) to look segments up in.
pub(crate) fn any() -> bool {
    drop(registry());
    HAS_UNITS.load(Ordering::Acquire)
}

/// The cached code for `key`, for a segment with `cache_len` decoder
/// entries, mapped and ready to install.
pub(crate) fn lookup(key: &CacheKey, cache_len: usize) -> Option<Compiled> {
    if !HAS_UNITS.load(Ordering::Acquire) {
        return None;
    }
    let guard = registry();
    let entry = guard.as_ref()?.units.get(key)?;
    let meta = &entry.meta;
    if meta.entries.len() != cache_len {
        return None;
    }
    let code = CodeMemory::linked(
        entry.code.clone(),
        meta.trampoline,
        &meta.entries,
        key.layout,
    )
    .ok()?;
    Some(Compiled {
        code,
        patch: meta.patch.clone(),
    })
}

/// Load cached JIT code from the shared object at `path` (written by
/// [`JitCacheOutput::write_shared_object`]) into the process-wide
/// registry, where every machine with `MachineOptions::jit_cache` finds it.
/// Returns the number of segments added; segments already registered keep
/// their code. Code from another build, host CPU or memory layout is never
/// used, but the object is loaded anyway (it stays loaded for good).
///
/// Loading a shared object runs native code from it: only load files you
/// would also run.
pub fn load_shared_object(path: impl AsRef<Path>) -> io::Result<usize> {
    let units = libriscv_jit_rt::load_shared_object(path.as_ref())?;
    let mut guard = registry();
    let registry = guard.get_or_insert_with(Registry::default);
    Ok(add(registry, units))
}

/// The number of segments with cached code in the registry: those
/// embedded in the program plus those from loaded shared objects.
pub fn registered_segments() -> usize {
    registry().as_ref().map_or(0, |r| r.units.len())
}

/// Collects the code the JIT compiles, to save it with
/// [`write_object`](Self::write_object) or
/// [`write_shared_object`](Self::write_shared_object) (see
/// `MachineOptions::jit_cache_output`). Clones share the same collection,
/// so several machines can record into one file. With `jit_background`,
/// segments are recorded when their compilation finishes: wait for it
/// ([`Machine::wait_for_jit`](crate::Machine::wait_for_jit)) before writing.
#[derive(Debug, Clone, Default)]
pub struct JitCacheOutput {
    units: Arc<Mutex<Vec<(CacheKey, Image)>>>,
}

impl PartialEq for JitCacheOutput {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.units, &other.units)
    }
}
impl Eq for JitCacheOutput {}

impl JitCacheOutput {
    pub fn new() -> Self {
        Self::default()
    }

    fn units(&self) -> MutexGuard<'_, Vec<(CacheKey, Image)>> {
        self.units.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The number of segments recorded.
    pub fn len(&self) -> usize {
        self.units().len()
    }

    pub fn is_empty(&self) -> bool {
        self.units().is_empty()
    }

    /// Record `image`, compiled for `key` (once per key).
    pub(crate) fn record(&self, key: CacheKey, image: &Image) {
        let mut units = self.units();
        if !units.iter().any(|(k, _)| *k == key) {
            units.push((key, image.clone()));
        }
    }

    /// The recorded code as an ELF relocatable object (x86-64).
    pub fn object(&self) -> io::Result<Vec<u8>> {
        write_elf(&self.units())
    }

    /// Write the recorded code as an ELF relocatable object, to link into a
    /// program: the program then finds it by itself (no registration).
    /// Link it as an object file, not from an archive, which would only
    /// pull it in to resolve a symbol. With `rustc`, for example:
    /// `cargo:rustc-link-arg=<path>` from a build script.
    pub fn write_object(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.object()?)
    }

    /// Write the recorded code as a shared object, for
    /// [`load_shared_object`]. Links the object with the system C compiler
    /// driver (`$CC`, else `cc`) and `libm` (for libcalls).
    pub fn write_shared_object(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let mut object = path.as_os_str().to_owned();
        object.push(".tmp.o");
        let object = std::path::PathBuf::from(object);
        self.write_object(&object)?;
        let cc = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
        let status = std::process::Command::new(&cc)
            .args(["-shared", "-nostdlib", "-Wl,-z,noexecstack", "-o"])
            .arg(path)
            .arg(&object)
            .arg("-lm")
            .status();
        let _ = std::fs::remove_file(&object);
        let status = status
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", cc.to_string_lossy())))?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "{} failed linking {}: {status}",
                cc.to_string_lossy(),
                path.display()
            )));
        }
        Ok(())
    }
}

// ELF constants (x86-64, relocatable).
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
const SHF_INFO_LINK: u64 = 0x40;
/// Never garbage-collect the section (`--gc-sections`).
const SHF_GNU_RETAIN: u64 = 0x20_0000;
const R_X86_64_GOTPCREL: u32 = 9;
const R_X86_64_PC64: u32 = 24;
const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_SECTION: u8 = 3;

/// Section indices of the object.
const S_TEXT: u16 = 1;
const S_META: u16 = 3;
const S_RECORDS: u16 = 4;
const S_TABLE: u16 = 6;
/// Symbol indices: the section symbols of text, metadata and records.
const SYM_TEXT: u32 = 1;
const SYM_META: u32 = 2;
const SYM_RECORDS: u32 = 3;
const SYM_TABLE: u32 = 4;
/// The first libcall symbol (after the table symbol).
const SYM_LIBCALLS: u32 = 5;

/// A relocation: `(offset, symbol, type, addend)`.
type Rela = (u64, u32, u32, i64);

fn invalid_data(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.to_owned())
}

/// Lay out `units` as an ELF relocatable object; see the module
/// documentation.
fn write_elf(units: &[(CacheKey, Image)]) -> io::Result<Vec<u8>> {
    if units.is_empty() {
        return Err(invalid_data("no compiled code to write"));
    }
    if !cfg!(target_arch = "x86_64") {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "saving JIT code is only supported on x86-64",
        ));
    }
    let mut text = Vec::new();
    let mut text_rela: Vec<Rela> = Vec::new();
    let mut meta = Vec::new();
    let mut records = Vec::new();
    let mut records_rela: Vec<Rela> = Vec::new();
    let mut libcalls: Vec<LibCall> = Vec::new();
    for (key, image) in units {
        text.resize(text.len().next_multiple_of(64), 0);
        let code = text.len() as u64;
        text.extend_from_slice(&image.bytes);
        for site in &image.libcalls {
            let index = match libcalls.iter().position(|&c| c == site.call) {
                Some(i) => i,
                None => {
                    libcall_symbol(site.call).ok_or_else(|| invalid_data("unknown libcall"))?;
                    libcalls.push(site.call);
                    libcalls.len() - 1
                }
            };
            text_rela.push((
                code + site.offset as u64,
                SYM_LIBCALLS + index as u32,
                R_X86_64_GOTPCREL,
                site.addend,
            ));
        }
        meta.resize(meta.len().next_multiple_of(8), 0);
        let m = metadata(key, image).ok_or_else(|| invalid_data("segment too large"))?;
        let meta_at = meta.len() as u64;
        meta.extend_from_slice(&m);
        let record = records.len() as u64;
        for w in [
            RECORD_MAGIC,
            0,
            image.bytes.len() as u64,
            0,
            m.len() as u64,
            0,
        ] {
            records.extend_from_slice(&w.to_le_bytes());
        }
        records_rela.push((record + 8, SYM_TEXT, R_X86_64_PC64, code as i64));
        records_rela.push((record + 24, SYM_META, R_X86_64_PC64, meta_at as i64));
    }
    let mut table = Vec::new();
    for w in [TABLE_MAGIC, units.len() as u64, 0] {
        table.extend_from_slice(&w.to_le_bytes());
    }
    let table_rela: Vec<Rela> = vec![(16, SYM_RECORDS, R_X86_64_PC64, 0)];
    debug_assert_eq!(records.len(), units.len() * RECORD_WORDS * 8);

    // Symbols and their names.
    let mut strtab = vec![0u8];
    let mut name = |s: &str| {
        let at = strtab.len() as u32;
        strtab.extend_from_slice(s.as_bytes());
        strtab.push(0);
        at
    };
    let mut symtab = vec![0u8; 24];
    let mut symbol = |name: u32, bind: u8, kind: u8, shndx: u16, size: u64| {
        symtab.extend_from_slice(&name.to_le_bytes());
        symtab.push(bind << 4 | kind);
        symtab.push(0); // STV_DEFAULT
        symtab.extend_from_slice(&shndx.to_le_bytes());
        symtab.extend_from_slice(&0u64.to_le_bytes());
        symtab.extend_from_slice(&size.to_le_bytes());
    };
    symbol(0, STB_LOCAL, STT_SECTION, S_TEXT, 0);
    symbol(0, STB_LOCAL, STT_SECTION, S_META, 0);
    symbol(0, STB_LOCAL, STT_SECTION, S_RECORDS, 0);
    // Weak: several objects can be linked into one program.
    let table_name = name(TABLE_SYMBOL);
    symbol(
        table_name,
        STB_WEAK,
        STT_OBJECT,
        S_TABLE,
        table.len() as u64,
    );
    for &call in &libcalls {
        let n = name(libcall_symbol(call).unwrap_or_default());
        symbol(n, STB_GLOBAL, STT_NOTYPE, 0, 0);
    }
    debug_assert_eq!(symtab.len() / 24, SYM_LIBCALLS as usize + libcalls.len());
    let rela = |r: &[Rela]| {
        let mut out = Vec::with_capacity(r.len() * 24);
        for &(offset, sym, kind, addend) in r {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(u64::from(sym) << 32 | u64::from(kind)).to_le_bytes());
            out.extend_from_slice(&addend.to_le_bytes());
        }
        out
    };

    struct Section {
        name: String,
        kind: u32,
        flags: u64,
        data: Vec<u8>,
        link: u32,
        info: u32,
        align: u64,
        entsize: u64,
    }
    let section = |name: &str, kind, flags, data, align| Section {
        name: name.into(),
        kind,
        flags,
        data,
        link: 0,
        info: 0,
        align,
        entsize: 0,
    };
    const SYMTAB: u32 = 9;
    let rela_section = |name: &str, target: u16, r: &[Rela]| Section {
        link: SYMTAB,
        info: u32::from(target),
        entsize: 24,
        ..section(name, SHT_RELA, SHF_INFO_LINK, rela(r), 8)
    };
    let retain = SHF_ALLOC | SHF_GNU_RETAIN;
    let mut sections = vec![
        section("", 0, 0, Vec::new(), 0),
        section(
            ".text.libriscv_jitcache",
            SHT_PROGBITS,
            retain | SHF_EXECINSTR,
            text,
            64,
        ),
        rela_section(".rela.text.libriscv_jitcache", S_TEXT, &text_rela),
        section(".rodata.libriscv_jitcache", SHT_PROGBITS, retain, meta, 8),
        section(SECTION, SHT_PROGBITS, retain, records, 8),
        rela_section(&format!(".rela{SECTION}"), S_RECORDS, &records_rela),
        section(
            ".rodata.libriscv_jitcache_table",
            SHT_PROGBITS,
            retain,
            table,
            8,
        ),
        rela_section(".rela.rodata.libriscv_jitcache_table", S_TABLE, &table_rela),
        // No executable stack.
        section(".note.GNU-stack", SHT_PROGBITS, 0, Vec::new(), 1),
        Section {
            link: SYMTAB + 1,
            // The first non-local symbol.
            info: SYM_TABLE,
            entsize: 24,
            ..section(".symtab", SHT_SYMTAB, 0, symtab, 8)
        },
        section(".strtab", SHT_STRTAB, 0, strtab, 1),
    ];
    debug_assert_eq!(sections.len() as u32, SYMTAB + 2);
    let mut shstrtab = vec![0u8];
    let mut names = Vec::with_capacity(sections.len() + 1);
    for s in &sections {
        if s.name.is_empty() {
            names.push(0);
        } else {
            names.push(shstrtab.len() as u32);
            shstrtab.extend_from_slice(s.name.as_bytes());
            shstrtab.push(0);
        }
    }
    names.push(shstrtab.len() as u32);
    shstrtab.extend_from_slice(b".shstrtab\0");
    sections.push(section(".shstrtab", SHT_STRTAB, 0, shstrtab, 1));

    // Layout: header, section contents, section headers.
    let mut out = vec![0u8; 64];
    let mut offsets = Vec::with_capacity(sections.len());
    for s in &sections {
        out.resize(out.len().next_multiple_of(s.align.max(1) as usize), 0);
        offsets.push(out.len() as u64);
        out.extend_from_slice(&s.data);
    }
    out.resize(out.len().next_multiple_of(8), 0);
    let shoff = out.len() as u64;
    for ((s, &offset), &name) in sections.iter().zip(&offsets).zip(&names) {
        out.extend_from_slice(&name.to_le_bytes());
        out.extend_from_slice(&s.kind.to_le_bytes());
        out.extend_from_slice(&s.flags.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // sh_addr
        let offset = if s.kind == 0 { 0 } else { offset };
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(s.data.len() as u64).to_le_bytes());
        out.extend_from_slice(&s.link.to_le_bytes());
        out.extend_from_slice(&s.info.to_le_bytes());
        out.extend_from_slice(&s.align.to_le_bytes());
        out.extend_from_slice(&s.entsize.to_le_bytes());
    }
    let mut header = Vec::with_capacity(64);
    header.extend_from_slice(b"\x7FELF");
    header.extend_from_slice(&[2, 1, 1, 0]); // 64-bit, little-endian, v1, SysV
    header.extend_from_slice(&[0; 8]);
    header.extend_from_slice(&1u16.to_le_bytes()); // ET_REL
    header.extend_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes()); // e_entry
    header.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
    header.extend_from_slice(&shoff.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    header.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    header.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
    header.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    header.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    header.extend_from_slice(&(sections.len() as u16).to_le_bytes());
    header.extend_from_slice(&((sections.len() - 1) as u16).to_le_bytes());
    if let Some(h) = out.get_mut(..64) {
        h.copy_from_slice(&header);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_tells_lengths_and_contents_apart() {
        assert_ne!(hash64(b""), hash64(b"\0"));
        assert_ne!(hash64(b"abcdefgh"), hash64(b"abcdefgi"));
        assert_ne!(hash64(&[0; 8]), hash64(&[0; 16]));
    }

    #[test]
    fn metadata_round_trips_and_truncation_is_refused() {
        let key = CacheKey {
            build: 1,
            vaddr: 0x10000,
            len: 64,
            crc32c: 0xDEAD_BEEF,
            hash: 7,
            execute_only: true,
            libc_fastpath: false,
            layout: MemoryLayout {
                mask: 0,
                read: (0x1000, 0x2000),
                write: (0x3000, 0x400),
            },
        };
        let image = Image {
            bytes: vec![0xC3; 16],
            trampoline: 8,
            entries: vec![CodeMemory::NO_ENTRY, 0, CodeMemory::NO_ENTRY, 4],
            patch: vec![1, 3],
            libcalls: Vec::new(),
            layout: key.layout,
        };
        let meta = metadata(&key, &image).unwrap();
        let (k, m) = parse(&meta).unwrap();
        assert_eq!(k, key);
        let expected = Meta {
            trampoline: 8,
            entries: image.entries.clone(),
            patch: image.patch.clone(),
        };
        assert_eq!(m, expected);
        for n in 0..meta.len() {
            assert!(parse(&meta[..n]).is_none(), "{n}");
        }
        // A patched entry without a function.
        let mut bad = image.clone();
        bad.patch = vec![2];
        assert!(parse(&metadata(&key, &bad).unwrap()).is_none());
    }
}
