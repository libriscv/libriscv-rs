//! Port of `tr_translate.cpp`, `tr_compiler.cpp` and the embedded and
//! `dlopen`ed translations of `tr_api.cpp`: ahead-of-time translation of
//! execute segments to safe Rust (C++ translates to C).
//!
//! A machine with [`MachineOptions::aot_output`](crate::MachineOptions::aot_output)
//! translates every execute segment it creates (its `emit` module). The translation
//! of a whole program is one Rust source file ([`AotOutput::source`]),
//! which can be used in two ways:
//!
//! * **Embedded**: compiled into the host program, where it is safe code
//!   like the rest (`forbid(unsafe_code)`), and registered with
//!   [`register`]:
//!
//!   ```ignore
//!   // At the crate root: the translation uses `become` and the
//!   // `preserve_none` calling convention, like libriscv's dispatch.
//!   #![feature(explicit_tail_calls, rust_preserve_none_cc)]
//!   #![allow(incomplete_features)]
//!
//!   include!("program.rs"); // defines `mod translation`
//!
//!   libriscv::aot::register(translation::META, translation::enter);
//!   ```
//!
//! * **Shared object**: [`AotOutput::write_shared_object`] compiles it with
//!   `rustc` into a `cdylib`, which [`load_shared_object`] (feature
//!   `aot-dylib`) loads at run time. The translated code is the same safe
//!   code; only the two exported functions at the boundary use `unsafe`,
//!   and so does loading it (in `libriscv-jit-rt`).
//!
//! Translations are keyed like the JIT cache: on the segment (address,
//! length, CRC32-C and a 64-bit hash of the code, execute-only, libc fast
//! path), the guest memory layout baked into the code, and a fingerprint
//! of the translator ([`fingerprint`]). When a machine with
//! [`MachineOptions::aot`](crate::MachineOptions::aot) (the default)
//! creates a segment that has a registered translation, the segment runs
//! it instead of being interpreted or JIT-compiled.
//!
//! Translated code behaves exactly like the JIT (it runs the same regions
//! of the same instructions, `codegen`): results and instruction counts
//! are the interpreter's, and it leaves to the interpreter at the same
//! places. Unlike the JIT, it doesn't depend on the host: the source can be
//! compiled for any 64-bit target.

mod emit;
mod structure;

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

pub(crate) use emit::Translation;

use crate::codegen::{NativeKey, region};
use crate::decoder_cache::DecodedSegment;
use crate::machine::Machine;
use crate::native_code::{MemoryLayout, NativeCode, NativeExit, NativeSegment};

/// First word of the metadata ("RVAOTM01").
const META_MAGIC: u64 = u64::from_le_bytes(*b"RVAOTM01");
/// The version of the interface between the host and translated code: the
/// metadata format and the signature of `enter`.
const ABI_VERSION: u64 = 2;

/// Declares [`X86_FEATURES`] and [`x86_detected`] from one list.
macro_rules! x86_features {
    ($($name:tt),* $(,)?) => {
        /// The x86-64 target features a translation may be compiled with
        /// (`-C target-cpu=native` for a shared object), by bit of the
        /// metadata's feature word: what the code may need the CPU to have.
        const X86_FEATURES: &[&str] = &[$($name),*];

        /// Per [`X86_FEATURES`] entry, whether this CPU has it.
        #[cfg(target_arch = "x86_64")]
        fn x86_detected() -> Vec<bool> {
            vec![$(std::is_x86_feature_detected!($name)),*]
        }
    };
}
x86_features!(
    "sse3",
    "ssse3",
    "sse4.1",
    "sse4.2",
    "popcnt",
    "avx",
    "avx2",
    "bmi1",
    "bmi2",
    "fma",
    "lzcnt",
    "f16c",
    "movbe",
    "aes",
    "pclmulqdq",
    "sha",
    "adx",
    "gfni",
    "vaes",
    "vpclmulqdq",
    "avx512f",
    "avx512bw",
    "avx512cd",
    "avx512dq",
    "avx512vl",
    "avx512vbmi",
    "avx512vnni",
);

/// The metadata's feature word as a Rust expression, evaluated where the
/// translation is compiled: the [`X86_FEATURES`] enabled there.
fn features_expr() -> String {
    let mut e = String::from("0");
    for (bit, name) in X86_FEATURES.iter().enumerate() {
        e.push_str(&format!(
            " | (cfg!(all(target_arch = \"x86_64\", target_feature = \"{name}\")) as u64) << {bit}"
        ));
    }
    e
}

/// The features of the feature word `features` this CPU lacks: a
/// translation needing them would die of an illegal instruction.
fn missing_features(features: u64) -> Vec<&'static str> {
    #[cfg(target_arch = "x86_64")]
    let detected = x86_detected();
    #[cfg(not(target_arch = "x86_64"))]
    let detected: Vec<bool> = Vec::new();
    let mut missing = Vec::new();
    for bit in 0..u64::BITS {
        if features >> bit & 1 == 0 {
            continue;
        }
        let n = bit as usize;
        if !detected.get(n).copied().unwrap_or(false) {
            missing.push(X86_FEATURES.get(n).copied().unwrap_or("unknown"));
        }
    }
    missing
}

/// The entry function of a translation (`translation::enter`):
/// `enter(segment, decoder_index, regs, fregs, arena, io)`, where `io` is
/// `[pc, counter, max, frm]` on entry and `[exit_kind, pc, counter, 0]` on
/// return. Returns false if the code can't run (unknown segment or entry,
/// arena too small), without having run anything.
pub type AotEnter =
    fn(usize, usize, &mut [u64; 32], &mut [u64; 32], &mut [u8], &mut [u64; 4]) -> bool;

/// Everything the code translated for a segment depends on besides the
/// segment itself and the memory layout: the libriscv version, its
/// extension features (which decide decoding and the decoder layout), and
/// the source of the translator and of what decides which instructions it
/// translates. Translations from another fingerprint are ignored.
pub fn fingerprint() -> u64 {
    static FINGERPRINT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let text = format!(
            "libriscv-aot {} {ABI_VERSION} {:?}",
            env!("CARGO_PKG_VERSION"),
            [
                cfg!(feature = "ext-a"),
                cfg!(feature = "ext-c"),
                cfg!(feature = "ext-f"),
                cfg!(feature = "ext-v"),
                cfg!(feature = "debug-checks"),
            ],
        );
        // What decides the translation, besides `codegen::SHARED_SOURCES`:
        // the translator and the metadata and `enter` protocol (this file).
        crate::codegen::fingerprint(
            text,
            &[
                include_str!("mod.rs"),
                include_str!("emit.rs"),
                include_str!("structure.rs"),
            ],
        )
    })
}

/// What a translation is looked up by: everything the code translated
/// for a segment depends on, with [`fingerprint`] as its build.
pub(crate) type AotKey = NativeKey;

/// The translation key of the segment `segment` describes.
pub(crate) fn key(segment: NativeKey) -> AotKey {
    segment.with_build(fingerprint())
}

/// How to enter a translation.
#[derive(Debug, Clone)]
enum Module {
    /// Compiled into this program.
    Embedded(AotEnter),
    /// In a loaded shared object.
    #[cfg(feature = "aot-dylib")]
    Shared(Arc<libriscv_jit_rt::aot::AotLibrary>),
}

/// The translated code of one execute segment.
#[derive(Debug)]
pub(crate) struct AotCode {
    module: Module,
    /// The segment's number in its translation.
    segment: usize,
    /// Bit `i`: native code can be entered at decoder entry `i`.
    entries: Arc<[u64]>,
}

impl AotCode {
    pub(crate) fn has_entry(&self, i: usize) -> bool {
        self.entries
            .get(i / 64)
            .is_some_and(|w| w >> (i % 64) & 1 != 0)
    }

    /// Run the translation from the block starting at `pc` (decoder entry
    /// `slot`), with the counter as it was *before* beginning that block.
    pub(crate) fn run(
        &self,
        m: &mut Machine,
        slot: usize,
        pc: u64,
        counter: u64,
        max: u64,
    ) -> NativeExit {
        let frm = u64::from(m.cpu.regs.fcsr().frm());
        let (regs, fregs) = m.cpu.regs.x_f_mut();
        let arena = m.memory.arena_mut();
        let mut io = [pc, counter, max, frm];
        let entered = match &self.module {
            Module::Embedded(enter) => enter(self.segment, slot, regs, fregs, arena, &mut io),
            #[cfg(feature = "aot-dylib")]
            Module::Shared(lib) => lib.enter(self.segment, slot, regs, fregs, arena, &mut io),
        };
        if !entered {
            return NativeExit::Unavailable;
        }
        let [kind, pc, counter, _] = io;
        NativeExit::from_raw(kind, pc, counter)
    }
}

/// A registered segment translation.
#[derive(Debug)]
struct Registered {
    module: Module,
    segment: usize,
    cache_len: usize,
    entries: Arc<[u64]>,
    patch: Arc<[usize]>,
}

/// The process-wide registry of translations.
static REGISTRY: Mutex<Option<HashMap<AotKey, Registered>>> = Mutex::new(None);
/// The registry has translations (checked without locking, before hashing
/// a segment for nothing).
static HAS_UNITS: AtomicBool = AtomicBool::new(false);

fn registry() -> MutexGuard<'static, Option<HashMap<AotKey, Registered>>> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Reads metadata words, refusing anything truncated.
struct Reader<'a>(&'a [u64]);

impl Reader<'_> {
    fn word(&mut self) -> Option<u64> {
        let (&w, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(w)
    }
    fn usize(&mut self) -> Option<usize> {
        usize::try_from(self.word()?).ok()
    }
    /// `n` words, which must be there.
    fn words(&mut self, n: usize) -> Option<&[u64]> {
        let (head, tail) = self.0.split_at_checked(n)?;
        self.0 = tail;
        Some(head)
    }
}

/// Add the segments described by `meta` to the registry, keeping existing
/// translations for a key. Returns the number added, or `None` if `meta`
/// isn't valid metadata or the code needs CPU features this one lacks.
fn add(meta: &[u64], module: &Module) -> Option<usize> {
    let mut r = Reader(meta);
    if r.word()? != META_MAGIC || r.word()? != ABI_VERSION {
        return None;
    }
    if !missing_features(r.word()?).is_empty() {
        return None;
    }
    let count = r.usize()?;
    let mut parsed = Vec::new();
    for segment in 0..count {
        let key: [u64; AotKey::WORDS] = r.words(AotKey::WORDS)?.try_into().ok()?;
        let key = AotKey::from_words(key)?;
        let cache_len = r.usize()?;
        let n = r.usize()?;
        let mut entries = vec![0u64; cache_len.div_ceil(64)];
        for &e in r.words(n)? {
            let e = usize::try_from(e).ok().filter(|&e| e < cache_len)?;
            *entries.get_mut(e / 64)? |= 1 << (e % 64);
        }
        let n = r.usize()?;
        let patch = r
            .words(n)?
            .iter()
            .map(|&p| usize::try_from(p).ok())
            .collect::<Option<Vec<_>>>()?;
        // Every patched entry must enter the code.
        if patch
            .iter()
            .any(|&p| entries.get(p / 64).is_none_or(|w| w >> (p % 64) & 1 == 0))
        {
            return None;
        }
        parsed.push((
            key,
            Registered {
                module: module.clone(),
                segment,
                cache_len,
                entries: entries.into(),
                patch: patch.into(),
            },
        ));
    }
    let mut guard = registry();
    let units = guard.get_or_insert_with(HashMap::new);
    let mut added = 0;
    for (key, reg) in parsed {
        if let std::collections::hash_map::Entry::Vacant(v) = units.entry(key) {
            v.insert(reg);
            added += 1;
        }
    }
    if !units.is_empty() {
        HAS_UNITS.store(true, Ordering::Release);
    }
    Some(added)
}

/// Register a translation compiled into this program (see the [module
/// documentation](self)): `meta` and `enter` are its `translation::META`
/// and `translation::enter`. Returns the number of segments added, 0 for
/// metadata from another version of libriscv's translator. Segments
/// already registered keep their translation. Translations made by
/// another build of libriscv (see [`fingerprint`]) are registered but
/// never used.
pub fn register(meta: &'static [u64], enter: AotEnter) -> usize {
    add(meta, &Module::Embedded(enter)).unwrap_or(0)
}

/// Load the translation compiled into the shared object at `path` (by
/// [`AotOutput::write_shared_object`]) into the process-wide registry.
/// Returns the number of segments added; segments already registered keep
/// their translation.
///
/// Loading a shared object runs native code from it: only load files you
/// would also run.
#[cfg(feature = "aot-dylib")]
pub fn load_shared_object(path: impl AsRef<Path>) -> io::Result<usize> {
    let lib = libriscv_jit_rt::aot::AotLibrary::load(path.as_ref())?;
    let meta = lib.meta().to_vec();
    if let [META_MAGIC, ABI_VERSION, features, ..] = meta.as_slice() {
        let missing = missing_features(*features);
        if !missing.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "the shared object was compiled for a CPU with features this one lacks: {}",
                    missing.join(", ")
                ),
            ));
        }
    }
    add(&meta, &Module::Shared(lib)).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "the shared object was translated by an incompatible version of libriscv",
        )
    })
}

/// The number of segments with a registered translation.
pub fn registered_segments() -> usize {
    registry().as_ref().map_or(0, HashMap::len)
}

/// True if any translation is registered.
pub(crate) fn any() -> bool {
    HAS_UNITS.load(Ordering::Acquire)
}

/// True if a translation is registered for `key`.
pub(crate) fn registered(key: &AotKey) -> bool {
    any()
        && registry()
            .as_ref()
            .is_some_and(|units| units.contains_key(key))
}

/// Install the registered translation for `key` in `exec` (a new segment,
/// before any native code). Returns false if there is none.
pub(crate) fn install(key: &AotKey, exec: &mut DecodedSegment) -> bool {
    if !any() {
        return false;
    }
    let found = {
        let guard = registry();
        guard
            .as_ref()
            .and_then(|units| units.get(key))
            .filter(|reg| reg.cache_len == exec.cache.len())
            .map(|reg| {
                (
                    AotCode {
                        module: reg.module.clone(),
                        segment: reg.segment,
                        entries: Arc::clone(&reg.entries),
                    },
                    reg.patch.to_vec(),
                )
            })
    };
    match found {
        Some((code, patch)) => {
            NativeSegment::install(exec, NativeCode::Aot(code), patch);
            true
        }
        None => false,
    }
}

/// Translate `exec` (a new segment, before any native code) for `layout`.
pub(crate) fn translate(exec: &DecodedSegment, layout: MemoryLayout) -> Option<Translation> {
    let analysis = region::analyse(exec);
    emit::translate(exec, &analysis, layout)
}

/// Collects the translations of the execute segments machines create, to
/// write them as Rust source ([`write_source`](Self::write_source)) or a
/// shared object ([`write_shared_object`](Self::write_shared_object)); see
/// `MachineOptions::aot_output`. Clones share the same collection, so
/// several machines can record into one translation.
#[derive(Debug, Clone, Default)]
pub struct AotOutput {
    units: Arc<Mutex<Vec<(AotKey, Translation)>>>,
}

impl PartialEq for AotOutput {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.units, &other.units)
    }
}
impl Eq for AotOutput {}

/// The prologue of the shared object written by
/// [`AotOutput::write_shared_object`]: the two exported functions, the only
/// `unsafe` code in it. The translation follows.
const SHARED_PROLOGUE: &str = r#"// A libriscv translation compiled as a shared object (`cdylib`), for
// `libriscv::aot::load_shared_object`. Generated: do not edit.
#![feature(explicit_tail_calls, rust_preserve_none_cc)]
#![allow(incomplete_features)]
#![deny(unsafe_code)]

/// Mirrors `libriscv_jit_rt::aot::RawCall`.
#[repr(C)]
pub struct RawCall {
    seg: u64,
    index: u64,
    regs: *mut u64,
    fregs: *mut u64,
    arena: *mut u8,
    arena_len: u64,
    io: [u64; 4],
}

/// Mirrors `libriscv_jit_rt::aot::RawMeta`.
#[repr(C)]
pub struct RawMeta {
    words: *const u64,
    len: u64,
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn libriscv_aot_meta() -> RawMeta {
    RawMeta { words: translation::META.as_ptr(), len: translation::META.len() as u64 }
}

/// # Safety
///
/// `call` points to a `RawCall` whose `regs` and `fregs` point to 32
/// registers each and whose `arena` points to `arena_len` bytes, all valid
/// and not otherwise accessed for the duration of the call
/// (`libriscv_jit_rt::aot::AotLibrary::enter` builds it from borrows).
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libriscv_aot_enter(call: *mut RawCall) -> u8 {
    // SAFETY: see the function's contract.
    let call = unsafe { &mut *call };
    // SAFETY: 32 registers, borrowed for the call (the contract).
    let regs = unsafe { &mut *call.regs.cast::<[u64; 32]>() };
    // SAFETY: 32 registers, borrowed for the call (the contract).
    let fregs = unsafe { &mut *call.fregs.cast::<[u64; 32]>() };
    // SAFETY: `arena_len` bytes, borrowed for the call (the contract).
    let arena = unsafe { std::slice::from_raw_parts_mut(call.arena, call.arena_len as usize) };
    u8::from(translation::enter(call.seg as usize, call.index as usize, regs, fregs, arena, &mut call.io))
}

"#;

impl AotOutput {
    pub fn new() -> Self {
        Self::default()
    }

    fn units(&self) -> MutexGuard<'_, Vec<(AotKey, Translation)>> {
        self.units.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The number of segments recorded.
    pub fn len(&self) -> usize {
        self.units().len()
    }

    pub fn is_empty(&self) -> bool {
        self.units().is_empty()
    }

    /// Record `translation`, made for `key` (once per key).
    pub(crate) fn record(&self, key: AotKey, translation: Translation) {
        let mut units = self.units();
        if !units.iter().any(|(k, _)| *k == key) {
            units.push((key, translation));
        }
    }

    /// The translation as Rust source: one module, `translation`, with
    /// `translation::META` and `translation::enter` for [`register`].
    /// Include it in a crate that enables `explicit_tail_calls` and
    /// `rust_preserve_none_cc` (see the [module documentation](self)).
    pub fn source(&self) -> String {
        use std::fmt::Write as _;
        let units = self.units();
        let mut o = String::new();
        let _ = writeln!(
            o,
            "// libriscv ahead-of-time translation: {} execute segments, translator {:016x}.
// Generated by libriscv {}: do not edit. Safe Rust; needs a crate with
// `#![feature(explicit_tail_calls, rust_preserve_none_cc)]`.
#[allow(warnings, clippy::all, clippy::pedantic, clippy::nursery, clippy::restriction)]
#[forbid(unsafe_code)]
pub mod translation {{
    const _: () = assert!(usize::BITS == 64, \"translations need a 64-bit target\");",
            units.len(),
            fingerprint(),
            env!("CARGO_PKG_VERSION"),
        );
        let _ = writeln!(o, "    mod rt {{{}    }}", emit::RUNTIME);
        let _ = writeln!(
            o,
            "    /// What `libriscv::aot::register` needs to use this translation."
        );
        let _ = write!(
            o,
            "    pub static META: &[u64] = &[{META_MAGIC:#x}, {ABI_VERSION},\n        {},\n        {},",
            features_expr(),
            units.len()
        );
        for (key, t) in units.iter() {
            o.push_str("\n        ");
            for w in key.words() {
                let _ = write!(o, "{w:#x}, ");
            }
            let _ = write!(o, "{}, {},", t.cache_len, t.entries.len());
            for (n, e) in t.entries.iter().enumerate() {
                if n % 16 == 0 {
                    o.push_str("\n        ");
                }
                let _ = write!(o, "{e}, ");
            }
            let _ = write!(o, "\n        {},", t.patch.len());
            for (n, p) in t.patch.iter().enumerate() {
                if n % 16 == 0 {
                    o.push_str("\n        ");
                }
                let _ = write!(o, "{p}, ");
            }
        }
        let _ = writeln!(o, "\n    ];");
        let _ = writeln!(
            o,
            "    /// The entry point, see `libriscv::aot::AotEnter`.
    pub fn enter(segment: usize, index: usize, x: &mut [u64; 32], f: &mut [u64; 32], arena: &mut [u8], io: &mut [u64; 4]) -> bool {{
        match segment {{"
        );
        for n in 0..units.len() {
            let _ = writeln!(o, "            {n} => s{n}::enter(index, x, f, arena, io),");
        }
        let _ = writeln!(o, "            _ => false,\n        }}\n    }}");
        for (n, (key, t)) in units.iter().enumerate() {
            let _ = writeln!(
                o,
                "    /// The segment at {:#x} ({} bytes).\n    mod s{n} {{",
                key.vaddr, key.len
            );
            o.push_str(&t.source);
            let _ = writeln!(o, "    }}");
        }
        let _ = writeln!(o, "}}");
        o
    }

    /// Write the translation as Rust source (see [`source`](Self::source)).
    pub fn write_source(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.source())
    }

    /// Compile the translation into a shared object for
    /// [`load_shared_object`], with `rustc` (`$RUSTC`, else `rustc`: a
    /// nightly with `explicit_tail_calls` and `rust_preserve_none_cc`, like
    /// libriscv's own toolchain, which rustup is asked for unless
    /// `$RUSTUP_TOOLCHAIN` says otherwise). Optimized for the host CPU
    /// (`-C target-cpu=native`): the shared object records the CPU features
    /// it was compiled with, and [`load_shared_object`] refuses it on a CPU
    /// without them. `$LIBRISCV_AOT_RUSTFLAGS` adds flags. The source is
    /// kept next to it, with `.rs` appended to its name.
    pub fn write_shared_object(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let mut source = path.as_os_str().to_owned();
        source.push(".rs");
        let source = std::path::PathBuf::from(source);
        let mut text = String::from(SHARED_PROLOGUE);
        text.push_str(&self.source());
        std::fs::write(&source, text)?;
        let rustc = std::env::var_os("RUSTC");
        let default = rustc.is_none();
        let rustc = rustc.unwrap_or_else(|| "rustc".into());
        let mut cmd = std::process::Command::new(&rustc);
        // The toolchain libriscv was built with (rustup's, when it ran the
        // build) has the features the translation needs; the default one,
        // outside this workspace, may well be a stable without them.
        if default
            && std::env::var_os("RUSTUP_TOOLCHAIN").is_none()
            && let Some(toolchain) = option_env!("RUSTUP_TOOLCHAIN")
        {
            cmd.env("RUSTUP_TOOLCHAIN", toolchain);
        }
        cmd.args([
            "--edition=2024",
            "--crate-type=cdylib",
            "--crate-name=libriscv_translation",
            "-Copt-level=3",
            "-Cpanic=abort",
            "-Ctarget-cpu=native",
            "-o",
        ])
        .arg(path)
        .arg(&source);
        if let Some(flags) = std::env::var_os("LIBRISCV_AOT_RUSTFLAGS") {
            cmd.args(flags.to_string_lossy().split_whitespace());
        }
        let status = cmd
            .status()
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", rustc.to_string_lossy())))?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "{} failed compiling {}: {status}",
                rustc.to_string_lossy(),
                source.display()
            )));
        }
        Ok(())
    }
}
