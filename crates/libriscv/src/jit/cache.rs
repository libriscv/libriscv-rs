//! Cache Cranelift JIT output across runs. Corresponds to C++'s translation
//! cache and embedded translations in `tr_translate.cpp`.
//!
//! A machine with a [`JitCacheOutput`] records the [`Image`] of every
//! segment it compiles. [`JitCacheOutput::write_object`] writes them as one
//! ELF relocatable object, to link into a host program (embedding), and
//! [`JitCacheOutput::write_shared_object`] links that into a shared object
//! with the system C compiler driver (`$CC`, else `cc`).
//!
//! Code linked into the program is found automatically, and shared objects
//! are added with [`load_shared_object`], both into one process-wide
//! registry. A [`CacheDir`] automates this for chosen segments, one shared
//! object per segment. When a machine creates an execute segment, it first looks for
//! code for it there (`MachineOptions::jit_cache`); only otherwise does the
//! JIT compile it. Code is keyed on the segment (address, length, contents,
//! execute-only, libc fast path), the memory layout compiled into the code
//! and a fingerprint of the build ([`build_fingerprint`](super::emit::build_fingerprint)),
//! so a file from another build, host CPU or memory layout is ignored, not
//! run.
//!
//! # Object format
//!
//! See `libriscv_jit_rt`'s `linked` module for the records and the table
//! that lead to each segment's code and metadata. The code is the image
//! as the JIT links it, with its libcalls (x86-64 only) as
//! `R_X86_64_GOTPCREL` relocations against the C library's functions. The metadata is
//! little-endian `u64`s: [`META_MAGIC`], the key (see [`NativeKey::words`]),
//! the trampoline offset, the entry count and the entries (`u32`, see
//! [`Image::entries`]), the patch count and the patched decoder
//! indices (`u32`), then the length and bytes of the guest code the unit
//! was compiled from.
//!
//! # Trust
//!
//! Key hashes are not cryptographic. Each unit stores its original guest code,
//! and installation requires an exact byte match to prevent collision-based
//! substitution. A [`CacheDir`] saves only code verified against host-selected
//! files, subject to [`MAX_SAVES`] and [`MAX_SAVE_BYTES`] per process. Cache
//! paths must be owned by the effective user and not writable by other users.
//! Loading native objects still requires trusting their code.

use std::collections::{HashMap, HashSet};
use std::fs::{DirBuilder, OpenOptions, Permissions};
use std::io::{self, Write};
use std::ops::Range;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use cranelift_codegen::ir::LibCall;
use libriscv_jit_rt::{
    CachedUnit, CodeMemory, LinkedCode, RECORD_MAGIC, RECORD_WORDS, SECTION, TABLE_MAGIC,
    TABLE_SYMBOL,
};

use super::emit::{Compiled, Image, build_fingerprint, libcall_symbol};
use crate::codegen::NativeKey;

/// First word of a unit's metadata ("RVJITM02"; units without their guest
/// code, "RVJITM01", are ignored).
const META_MAGIC: u64 = u64::from_le_bytes(*b"RVJITM02");

/// What cached code is looked up by: everything the code compiled for a
/// segment depends on, with [`build_fingerprint`] as its build.
pub(crate) type CacheKey = NativeKey;

/// The cache key of the segment `segment` describes, or `None` where the
/// JIT is not supported.
pub(crate) fn cache_key(segment: NativeKey) -> Option<CacheKey> {
    Some(segment.with_build(build_fingerprint()?))
}

fn write_key(key: &CacheKey, out: &mut Vec<u8>) {
    for w in key.words() {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

fn read_key(r: &mut Reader<'_>) -> Option<CacheKey> {
    let mut w = [0u64; NativeKey::WORDS];
    for x in &mut w {
        *x = r.u64()?;
    }
    NativeKey::from_words(w)
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

/// The metadata of `image`, compiled from the guest code `code` for `key`.
fn metadata(key: &CacheKey, image: &Image, code: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&META_MAGIC.to_le_bytes());
    write_key(key, &mut out);
    out.extend_from_slice(&(image.trampoline as u64).to_le_bytes());
    out.extend_from_slice(&(image.entries.len() as u64).to_le_bytes());
    for &e in &image.entries {
        out.extend_from_slice(&e.to_le_bytes());
    }
    out.extend_from_slice(&(image.patch.len() as u64).to_le_bytes());
    for &p in &image.patch {
        out.extend_from_slice(&u32::try_from(p).ok()?.to_le_bytes());
    }
    out.extend_from_slice(&(code.len() as u64).to_le_bytes());
    out.extend_from_slice(code);
    Some(out)
}

/// A unit's metadata, besides its key.
#[derive(Debug, PartialEq, Eq)]
struct Meta {
    trampoline: usize,
    entries: Vec<u32>,
    patch: Vec<usize>,
    /// The guest code the unit was compiled from.
    code: Vec<u8>,
}

impl Meta {
    /// True if the unit was compiled from exactly `code`, a segment with
    /// `cache_len` decoder entries: only then is it the code the JIT would
    /// compile for the segment. (The key's hashes alone can be forged.)
    fn fits(&self, code: &[u8], cache_len: usize) -> bool {
        self.entries.len() == cache_len && self.code == code
    }
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
    let key = read_key(&mut r)?;
    let trampoline = usize::try_from(r.u64()?).ok()?;
    let n = r.count()?;
    let entries = (0..n).map(|_| r.u32()).collect::<Option<Vec<_>>>()?;
    let n = r.count()?;
    let patch = (0..n)
        .map(|_| r.u32().map(|p| p as usize))
        .collect::<Option<Vec<_>>>()?;
    let n = usize::try_from(r.u64()?).ok()?;
    let code = r.take(n)?.to_vec();
    if code.len() as u64 != key.len || !r.0.is_empty() {
        return None;
    }
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
            code,
        },
    ))
}

/// The process-wide registry of cached code.
#[derive(Debug, Default)]
struct Registry {
    /// The program's embedded code has been added.
    embedded: bool,
    /// Units with the same key but other code (at most one of them can
    /// be the segment's) are all kept: the segment's code picks one.
    units: HashMap<CacheKey, Vec<Entry>>,
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

/// Add `units` to the registry, keeping existing code for a key and
/// guest code (as C++ does). Returns the number of units added.
fn add(registry: &mut Registry, units: Vec<CachedUnit>) -> usize {
    let mut added = 0;
    for unit in units {
        let Some((key, meta)) = parse(&unit.meta) else {
            continue;
        };
        let same_key = registry.units.entry(key).or_default();
        if !same_key.iter().any(|e| e.meta.code == meta.code) {
            same_key.push(Entry {
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

/// The cached code for `key`, for the segment of guest code `code` with
/// `cache_len` decoder entries, mapped and ready to install. Only code
/// compiled from exactly `code` is used.
pub(crate) fn lookup(key: &CacheKey, code: &[u8], cache_len: usize) -> Option<Compiled> {
    if !HAS_UNITS.load(Ordering::Acquire) {
        return None;
    }
    let guard = registry();
    let entry = guard
        .as_ref()?
        .units
        .get(key)?
        .iter()
        .find(|e| e.meta.fits(code, cache_len))?;
    let meta = &entry.meta;
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

/// Load a shared object produced by [`JitCacheOutput::write_shared_object`]
/// into the process-wide cache registry. Returns the number of added segments;
/// existing entries retain their code. Machines use only matching builds,
/// CPUs, and layouts, but the object remains loaded even if no entry matches.
///
/// Loading executes native code. Only load trusted objects.
pub fn load_shared_object(path: impl AsRef<Path>) -> io::Result<usize> {
    let units = libriscv_jit_rt::load_shared_object(path.as_ref())?;
    let mut guard = registry();
    let registry = guard.get_or_insert_with(Registry::default);
    Ok(add(registry, units))
}

/// The number of segments with cached code in the registry: those
/// embedded in the program plus those from loaded shared objects.
pub fn registered_segments() -> usize {
    registry()
        .as_ref()
        .map_or(0, |r| r.units.values().map(Vec::len).sum())
}

/// Directory cache for selected execute segments (`MachineOptions::jit_cache_dir`).
/// With `jit`, a matching object is loaded if present; otherwise the segment is
/// compiled and saved through `$CC` or `cc`. Files are grouped by JIT build
/// fingerprint and retained indefinitely. Stable code at fixed addresses,
/// such as the dynamic linker and system libraries, benefits most.
///
/// Saving requires exact bytes from a host-selected file: the initial ELF
/// when [`initial_segment`](Self::initial_segment) is enabled, or a segment
/// inside [`ranges`](Self::ranges) verified against a pinned host-file mapping.
/// Files are re-read for comparison. Rebuilt segments are excluded.
///
/// Per-process limits are [`MAX_SAVES`] files and [`MAX_SAVE_BYTES`] bytes.
/// Build directories with [`MAX_DIR_FILES`] entries receive no new files.
/// Loading also requires an exact match with the stored original guest bytes.
///
/// Cache directories and files must belong to the effective user and must not
/// be writable by group or others. Created directories use mode 0700.
/// Loading objects executes native code; cache paths must remain trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheDir {
    pub path: PathBuf,
    /// Cache the machine's initial execute segment (the one created from
    /// the ELF program the machine was created with). Only for a program
    /// the host chose, not one a guest can name (like an ELF's
    /// `PT_INTERP`).
    pub initial_segment: bool,
    /// Cache the segments lying entirely inside one of these address
    /// ranges (saved only if their bytes come from a pinned file, see
    /// above).
    pub ranges: Vec<Range<u64>>,
}

/// The most cache files one process writes to [`CacheDir`]s: a guest
/// that keeps creating segments must not fill the disk (nor run the
/// linker for each). `rvlinux` needs one for the dynamic linker and one
/// per pinned library it uses.
pub const MAX_SAVES: usize = 32;
/// The most bytes (native code plus the guest code stored with it) one
/// process saves to [`CacheDir`]s. `libc.so`'s native code is about 36 MB.
pub const MAX_SAVE_BYTES: u64 = 512 << 20;
/// A [`CacheDir`]'s build subdirectory with this many entries gets no more
/// files.
pub const MAX_DIR_FILES: usize = 256;

/// Files saved, and their bytes, by this process (for [`MAX_SAVES`] and
/// [`MAX_SAVE_BYTES`]).
static SAVED: Mutex<(usize, u64)> = Mutex::new((0, 0));

/// True if this process may still save a file to a [`CacheDir`].
pub(crate) fn may_save() -> bool {
    SAVED.lock().unwrap_or_else(PoisonError::into_inner).0 < MAX_SAVES
}

/// Count a save of `bytes` against the limits; false (counting nothing)
/// if over them.
fn reserve_save(bytes: u64) -> bool {
    let mut saved = SAVED.lock().unwrap_or_else(PoisonError::into_inner);
    let total = saved.1.saturating_add(bytes);
    if saved.0 >= MAX_SAVES || total > MAX_SAVE_BYTES {
        return false;
    }
    *saved = (saved.0 + 1, total);
    true
}

impl CacheDir {
    /// True if the segment of `len` bytes at `vaddr` is looked up here.
    /// (Saving it takes more, see the type's documentation.)
    pub(crate) fn covers(&self, vaddr: u64, len: u64, initial: bool) -> bool {
        (initial && self.initial_segment) || self.in_ranges(vaddr, len)
    }

    /// True if the segment of `len` bytes at `vaddr` lies in one of
    /// [`ranges`](Self::ranges).
    pub(crate) fn in_ranges(&self, vaddr: u64, len: u64) -> bool {
        vaddr
            .checked_add(len)
            .is_some_and(|end| self.ranges.iter().any(|r| r.start <= vaddr && end <= r.end))
    }

    /// The file holding the code for `key`.
    pub(crate) fn file(&self, key: &CacheKey) -> PathBuf {
        let words: Vec<u8> = key.words().iter().flat_map(|w| w.to_le_bytes()).collect();
        self.path.join(format!("{:016x}", key.build)).join(format!(
            "{:x}-{:016x}.so",
            key.vaddr,
            crate::codegen::hash64(&words)
        ))
    }
}

fn untrusted(path: &Path, why: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!("{} {why}", path.display()),
    )
}

/// Ok if `meta` (of `path`) belongs to this process's user and nobody
/// else can write to it.
fn check_private(path: &Path, meta: &std::fs::Metadata) -> io::Result<()> {
    if meta.uid() != libriscv_jit_rt::effective_uid() {
        return Err(untrusted(path, "is owned by another user"));
    }
    if meta.mode() & 0o022 != 0 {
        return Err(untrusted(path, "is writable by group or others"));
    }
    Ok(())
}

/// Ok if the cache file `file` (`<dir>/<build>/<name>`), its build
/// subdirectory and its cache directory are private to this process's
/// user (see [`check_private`]). Only the cache directory may be a
/// symbolic link; with `file_too` false, `file` needn't exist.
fn check_cache_path(file: &Path, file_too: bool) -> io::Result<()> {
    let build = file
        .parent()
        .ok_or_else(|| untrusted(file, "has no directory"))?;
    let root = build
        .parent()
        .ok_or_else(|| untrusted(build, "has no parent"))?;
    let meta = std::fs::metadata(root)?;
    if !meta.is_dir() {
        return Err(untrusted(root, "is not a directory"));
    }
    check_private(root, &meta)?;
    let meta = std::fs::symlink_metadata(build)?;
    if !meta.is_dir() {
        return Err(untrusted(build, "is not a directory"));
    }
    check_private(build, &meta)?;
    if file_too {
        let meta = std::fs::symlink_metadata(file)?;
        if !meta.is_file() {
            return Err(untrusted(file, "is not a regular file"));
        }
        check_private(file, &meta)?;
    }
    Ok(())
}

/// Create `dir` and its missing parents with mode 0700.
fn create_private_dir(dir: &Path) -> io::Result<()> {
    DirBuilder::new().recursive(true).mode(0o700).create(dir)
}

/// A hard to guess number, different on each call: names of temporary
/// files nobody can plant in advance.
fn unpredictable() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    static N: AtomicU64 = AtomicU64::new(0);
    // Randomly keyed by the standard library (from the OS).
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(N.fetch_add(1, Ordering::Relaxed));
    h.write_u32(std::process::id());
    if let Ok(t) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        h.write_u128(t.as_nanos());
    }
    h.finish()
}

/// A directory of this process's own, with mode 0700 and an unpredictable
/// name, created inside `parent` (never an existing one) and removed with
/// what's in it when dropped. Temporary files go in it: nobody else can
/// plant files or symbolic links there.
#[derive(Debug)]
struct TempDir(PathBuf);

impl TempDir {
    fn new(parent: &Path) -> io::Result<Self> {
        let mut last = None;
        for _ in 0..8 {
            let path = parent.join(format!(".libriscv-tmp-{:016x}", unpredictable()));
            match DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| io::Error::from(io::ErrorKind::AlreadyExists)))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write `data` to the new file `path` (`O_CREAT | O_EXCL`: never an
/// existing file or a symbolic link), with mode 0600.
fn write_new(path: &Path, data: &[u8]) -> io::Result<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?
        .write_all(data)
}

/// The cache files this process has tried loading (once each).
static LOADED_FILES: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);

/// Load the cache file `file` into the registry, if it exists, this
/// process has not tried before, and it and its directories are private to
/// this process's user.
pub(crate) fn load_file(file: &Path, verbose: bool) {
    if !file.exists() {
        return;
    }
    {
        let mut loaded = LOADED_FILES.lock().unwrap_or_else(PoisonError::into_inner);
        if !loaded
            .get_or_insert_with(HashSet::new)
            .insert(file.to_owned())
        {
            return;
        }
    }
    let loaded = check_cache_path(file, true).and_then(|()| load_shared_object(file));
    match loaded {
        Ok(n) if verbose => eprintln!(
            "libriscv: Loaded cached JIT code for {n} segments from {}",
            file.display()
        ),
        Ok(_) => {}
        Err(e) => {
            if verbose {
                eprintln!("libriscv: Could not load {}: {e}", file.display());
            }
        }
    }
}

/// Write `image`, compiled from the guest code `code` for `key`, to the
/// cache file `file`, within the limits on saves ([`MAX_SAVES`],
/// [`MAX_SAVE_BYTES`], [`MAX_DIR_FILES`]). Linked in a private temporary
/// directory and renamed into place, so other processes never see a
/// partial file.
pub(crate) fn save_file(key: CacheKey, image: &Image, code: &[u8], file: &Path, verbose: bool) {
    match try_save_file(key, image, code, file) {
        Ok(()) if verbose => eprintln!(
            "libriscv: Saved JIT code for the segment at 0x{:X} to {}",
            key.vaddr,
            file.display()
        ),
        Ok(()) => {}
        Err(e) => {
            if verbose {
                eprintln!("libriscv: Could not save {}: {e}", file.display());
            }
        }
    }
}

fn try_save_file(key: CacheKey, image: &Image, code: &[u8], file: &Path) -> io::Result<()> {
    let dir = file
        .parent()
        .ok_or_else(|| untrusted(file, "has no directory"))?;
    let bytes = (image.bytes.len() as u64).saturating_add(code.len() as u64);
    if !reserve_save(bytes) {
        return Err(io::Error::other(format!(
            "this process has saved the most it may ({MAX_SAVES} files, {MAX_SAVE_BYTES} bytes)"
        )));
    }
    create_private_dir(dir)?;
    check_cache_path(file, false)?;
    if std::fs::read_dir(dir)?.take(MAX_DIR_FILES).count() >= MAX_DIR_FILES {
        return Err(io::Error::other(format!(
            "{} has {MAX_DIR_FILES} entries already",
            dir.display()
        )));
    }
    let output = JitCacheOutput::new();
    output.record(key, image, code);
    let tmp = TempDir::new(dir)?;
    let so = tmp.0.join("unit.so");
    output.link_shared_object(&tmp.0, &so)?;
    std::fs::set_permissions(&so, Permissions::from_mode(0o600))?;
    // Replaces whatever is at `file`, a symbolic link too, never
    // following it.
    std::fs::rename(&so, file)
}

/// Cache files being written by background compilations.
static PENDING_SAVES: Mutex<usize> = Mutex::new(0);
static SAVES_DONE: Condvar = Condvar::new();

/// Counts a background compilation that will write a cache file, until
/// dropped.
pub(crate) struct PendingSave(());

impl PendingSave {
    pub(crate) fn new() -> Self {
        *PENDING_SAVES.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        Self(())
    }
}

impl Drop for PendingSave {
    fn drop(&mut self) {
        let mut n = PENDING_SAVES.lock().unwrap_or_else(PoisonError::into_inner);
        *n = n.saturating_sub(1);
        SAVES_DONE.notify_all();
    }
}

/// The number of background compilations that are going to write a file
/// to a [`CacheDir`].
pub fn pending_saves() -> usize {
    *PENDING_SAVES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Wait until every background compilation that writes to a [`CacheDir`]
/// has written its file. Background compilations do not keep the process
/// alive: call this before exiting so that their code is saved.
pub fn wait_for_saves() {
    let mut n = PENDING_SAVES.lock().unwrap_or_else(PoisonError::into_inner);
    while *n > 0 {
        n = SAVES_DONE.wait(n).unwrap_or_else(PoisonError::into_inner);
    }
}

/// A segment's code as the JIT compiled it, with the guest code it was
/// compiled from.
#[derive(Debug)]
struct Unit {
    key: CacheKey,
    image: Image,
    code: Vec<u8>,
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
    units: Arc<Mutex<Vec<Unit>>>,
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

    fn units(&self) -> MutexGuard<'_, Vec<Unit>> {
        self.units.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The number of segments recorded.
    pub fn len(&self) -> usize {
        self.units().len()
    }

    pub fn is_empty(&self) -> bool {
        self.units().is_empty()
    }

    /// Record `image`, compiled from the guest code `code` for `key` (once
    /// per key and code).
    pub(crate) fn record(&self, key: CacheKey, image: &Image, code: &[u8]) {
        let mut units = self.units();
        if !units.iter().any(|u| u.key == key && u.code == code) {
            units.push(Unit {
                key,
                image: image.clone(),
                code: code.to_vec(),
            });
        }
    }

    /// The recorded code as an ELF relocatable object for the host (x86-64
    /// or AArch64).
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
    /// driver (`$CC`, else `cc`) and `libm` (for libcalls). The object is
    /// written to a private temporary directory next to `path`.
    pub fn write_shared_object(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let tmp = TempDir::new(parent)?;
        self.link_shared_object(&tmp.0, path)
    }

    /// [`write_shared_object`](Self::write_shared_object) to `path`, with
    /// the object in the private directory `tmp` (a [`TempDir`]).
    fn link_shared_object(&self, tmp: &Path, path: &Path) -> io::Result<()> {
        let object = tmp.join("unit.o");
        write_new(&object, &self.object()?)?;
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

// ELF constants (relocatable).
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
/// The host's machine type and its 64-bit PC-relative relocation (`S + A -
/// P`): x86-64, or AArch64, where generated code has no libcalls (Cranelift
/// rounds and fuses natively), so it needs no GOT relocations.
#[cfg(target_arch = "aarch64")]
const HOST: Option<(u16, u32)> = Some((183, 260)); // EM_AARCH64, R_AARCH64_PREL64
#[cfg(target_arch = "x86_64")]
const HOST: Option<(u16, u32)> = Some((62, 24)); // EM_X86_64, R_X86_64_PC64
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const HOST: Option<(u16, u32)> = None;
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
fn write_elf(units: &[Unit]) -> io::Result<Vec<u8>> {
    if units.is_empty() {
        return Err(invalid_data("no compiled code to write"));
    }
    let unsupported = || {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "saving JIT code is only supported on x86-64 and AArch64",
        )
    };
    let (machine, r_pc64) = HOST.ok_or_else(unsupported)?;
    let mut text = Vec::new();
    let mut text_rela: Vec<Rela> = Vec::new();
    let mut meta = Vec::new();
    let mut records = Vec::new();
    let mut records_rela: Vec<Rela> = Vec::new();
    let mut libcalls: Vec<LibCall> = Vec::new();
    for Unit {
        key,
        image,
        code: guest,
    } in units
    {
        text.resize(text.len().next_multiple_of(64), 0);
        let code = text.len() as u64;
        text.extend_from_slice(&image.bytes);
        if !image.libcalls.is_empty() && !cfg!(target_arch = "x86_64") {
            return Err(unsupported());
        }
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
        let m = metadata(key, image, guest).ok_or_else(|| invalid_data("segment too large"))?;
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
        records_rela.push((record + 8, SYM_TEXT, r_pc64, code as i64));
        records_rela.push((record + 24, SYM_META, r_pc64, meta_at as i64));
    }
    let mut table = Vec::new();
    for w in [TABLE_MAGIC, units.len() as u64, 0] {
        table.extend_from_slice(&w.to_le_bytes());
    }
    let table_rela: Vec<Rela> = vec![(16, SYM_RECORDS, r_pc64, 0)];
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
    header.extend_from_slice(&machine.to_le_bytes());
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
    use libriscv_jit_rt::MemoryLayout;

    use super::*;

    fn key() -> CacheKey {
        CacheKey {
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
                unlimited: false,
                unguarded: false,
                paged: false,
            },
        }
    }

    fn image(key: &CacheKey) -> Image {
        Image {
            bytes: vec![0xC3; 16],
            trampoline: 8,
            entries: vec![CodeMemory::NO_ENTRY, 0, CodeMemory::NO_ENTRY, 4],
            patch: vec![1, 3],
            libcalls: Vec::new(),
            layout: key.layout,
        }
    }

    /// 64 bytes of guest code (the key's length).
    fn guest_code() -> Vec<u8> {
        (0..64u8).collect()
    }

    #[test]
    fn metadata_round_trips_and_truncation_is_refused() {
        let key = key();
        let image = image(&key);
        let code = guest_code();
        let meta = metadata(&key, &image, &code).unwrap();
        let (k, m) = parse(&meta).unwrap();
        assert_eq!(k, key);
        let expected = Meta {
            trampoline: 8,
            entries: image.entries.clone(),
            patch: image.patch.clone(),
            code: code.clone(),
        };
        assert_eq!(m, expected);
        for n in 0..meta.len() {
            assert!(parse(&meta[..n]).is_none(), "{n}");
        }
        // Trailing bytes.
        let mut long = meta.clone();
        long.push(0);
        assert!(parse(&long).is_none());
        // A patched entry without a function.
        let mut bad = image.clone();
        bad.patch = vec![2];
        assert!(parse(&metadata(&key, &bad, &code).unwrap()).is_none());
        // Guest code of another length than the key's.
        assert!(parse(&metadata(&key, &image, &code[..60]).unwrap()).is_none());
        // Units of the first format (without the guest code).
        let mut old = meta.clone();
        old[..8].copy_from_slice(b"RVJITM01");
        assert!(parse(&old).is_none());
    }

    /// The key's CRC32-C and hash can be forged (see
    /// `decoder_cache::tests::forge_collision`): a unit is only used for
    /// a segment with exactly the bytes it was compiled from, whatever its
    /// key says.
    #[test]
    fn cached_code_is_only_used_for_the_code_it_was_compiled_from() {
        let key = key();
        let image = image(&key);
        let code = guest_code();
        let (_, meta) = parse(&metadata(&key, &image, &code).unwrap()).unwrap();
        assert!(meta.fits(&code, image.entries.len()));
        let mut forged = code.clone();
        forged[17] ^= 0x40;
        assert!(!meta.fits(&forged, image.entries.len()));
        assert!(!meta.fits(&code[..60], image.entries.len()));
        assert!(!meta.fits(&code, image.entries.len() + 1));
    }

    #[test]
    fn saves_are_limited() {
        let mut saved = SAVED.lock().unwrap_or_else(PoisonError::into_inner);
        let before = *saved;
        *saved = (0, 0);
        drop(saved);
        assert!(may_save());
        assert!(!reserve_save(MAX_SAVE_BYTES + 1));
        assert!(reserve_save(MAX_SAVE_BYTES - 10));
        assert!(!reserve_save(11));
        assert!(reserve_save(10));
        *SAVED.lock().unwrap() = (MAX_SAVES - 1, 0);
        assert!(may_save());
        assert!(reserve_save(0));
        assert!(!may_save());
        assert!(!reserve_save(0));
        *SAVED.lock().unwrap() = before;
    }

    /// A fresh directory for `test` in the system's temporary directory,
    /// created with `mode`.
    fn scratch(test: &str, mode: u32) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "libriscv-jit-cache-{test}-{}-{:x}",
            std::process::id(),
            unpredictable()
        ));
        DirBuilder::new().mode(mode).create(&dir).unwrap();
        std::fs::set_permissions(&dir, Permissions::from_mode(mode)).unwrap();
        dir
    }

    fn set_mode(path: &Path, mode: u32) {
        std::fs::set_permissions(path, Permissions::from_mode(mode)).unwrap();
    }

    /// Cache files are only loaded from (and saved to) directories and
    /// files of this user that nobody else can write to.
    #[test]
    fn cache_paths_must_be_private() {
        let root = scratch("private", 0o700);
        let build = root.join("0123456789abcdef");
        create_private_dir(&build).unwrap();
        assert_eq!(std::fs::metadata(&build).unwrap().mode() & 0o777, 0o700);
        let file = build.join("10000-0123456789abcdef.so");
        // Saving needs no file yet; loading does.
        check_cache_path(&file, false).unwrap();
        assert!(check_cache_path(&file, true).is_err());
        write_new(&file, b"code").unwrap();
        check_cache_path(&file, true).unwrap();

        set_mode(&file, 0o620);
        assert!(check_cache_path(&file, true).is_err());
        set_mode(&file, 0o602);
        assert!(check_cache_path(&file, true).is_err());
        set_mode(&file, 0o644);
        check_cache_path(&file, true).unwrap();

        set_mode(&build, 0o770);
        assert!(check_cache_path(&file, true).is_err());
        assert!(check_cache_path(&file, false).is_err());
        set_mode(&build, 0o755);
        check_cache_path(&file, true).unwrap();

        set_mode(&root, 0o777);
        assert!(check_cache_path(&file, true).is_err());
        assert!(check_cache_path(&file, false).is_err());
        set_mode(&root, 0o1777);
        assert!(check_cache_path(&file, true).is_err());
        set_mode(&root, 0o700);

        // A symbolic link in place of the file or the build directory.
        let elsewhere = root.join("elsewhere.so");
        write_new(&elsewhere, b"code").unwrap();
        std::fs::remove_file(&file).unwrap();
        std::os::unix::fs::symlink(&elsewhere, &file).unwrap();
        assert!(check_cache_path(&file, true).is_err());
        std::fs::remove_file(&file).unwrap();
        std::fs::remove_dir(&build).unwrap();
        std::os::unix::fs::symlink(&root, &build).unwrap();
        assert!(check_cache_path(&file, false).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Temporary files never reuse an existing name, nor follow a
    /// symbolic link planted in their place.
    #[test]
    fn temporary_files_are_new() {
        let root = scratch("temp", 0o700);
        let victim = root.join("victim");
        std::fs::write(&victim, b"precious").unwrap();
        let planted = root.join("planted");
        std::os::unix::fs::symlink(&victim, &planted).unwrap();
        assert!(write_new(&planted, b"overwritten").is_err());
        assert!(write_new(&victim, b"overwritten").is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        let new = root.join("new");
        write_new(&new, b"data").unwrap();
        assert_eq!(std::fs::metadata(&new).unwrap().mode() & 0o777, 0o600);

        // Private temporary directories, each new, removed when dropped.
        let a = TempDir::new(&root).unwrap();
        let b = TempDir::new(&root).unwrap();
        assert_ne!(a.0, b.0);
        assert_eq!(std::fs::metadata(&a.0).unwrap().mode() & 0o777, 0o700);
        write_new(&a.0.join("unit.o"), b"object").unwrap();
        let path = a.0.clone();
        drop(a);
        assert!(!path.exists());
        drop(b);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
