//! Cached JIT code linked into the program or into a shared object.
//!
//! `libriscv` saves the code it compiled for an execute segment as an ELF
//! relocatable object (its `jit_cache` module writes it, see there for the
//! format). The object has one *record* per segment in the section
//! [`SECTION`], and a *table* of its records behind the symbol
//! [`TABLE_SYMBOL`]. All of them are `u64` words, and the addresses in them
//! are relative (`R_X86_64_PC64`), so the sections need no run-time
//! relocations and can be read-only:
//!
//! * record (`RECORD_WORDS`): `[RECORD_MAGIC, code - &word[1], code_len,
//!   meta - &word[3], meta_len, 0]`. `code` is the native code, `meta` what
//!   `libriscv` needs to use it (its key, entry table and so on).
//! * table (`TABLE_WORDS`): `[TABLE_MAGIC, count, records - &word[2]]`.
//!
//! Objects linked into the program are found through the linker-defined
//! `__start_`/`__stop_` symbols of [`SECTION`] (every record of every such
//! object, in one array), shared objects through their table (`dlsym`).

use std::io;
use std::path::Path;
use std::sync::Arc;

/// Name of the section holding the records (a C identifier, so that
/// linkers define `__start_`/`__stop_` symbols for it).
pub const SECTION: &str = "libriscv_jitcache";
/// The symbol of a shared object's record table.
pub const TABLE_SYMBOL: &str = "libriscv_jitcache_table";
/// First word of a record ("RVJITC01", little-endian).
pub const RECORD_MAGIC: u64 = u64::from_le_bytes(*b"RVJITC01");
/// First word of a table ("RVJITT01", little-endian).
pub const TABLE_MAGIC: u64 = u64::from_le_bytes(*b"RVJITT01");
/// Words per record.
pub const RECORD_WORDS: usize = 6;
/// Words in a table.
pub const TABLE_WORDS: usize = 3;

/// Refuse sizes beyond this: no real segment's code or metadata comes close.
const MAX_LEN: u64 = 1 << 36;

/// A shared object loaded with `dlopen`, closed when the last
/// [`LinkedCode`] from it is dropped.
#[derive(Debug)]
pub struct SharedObject {
    /// The `dlopen` handle, as an address (so that this is `Send + Sync`:
    /// the handle is only ever passed back to `dlclose`).
    handle: usize,
}

impl Drop for SharedObject {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        // SAFETY: `handle` came from a successful `dlopen` and is closed
        // exactly once, here, after the last code from it is gone: every
        // `CodeMemory` running it holds this object through an `Arc`.
        unsafe {
            sys::dlclose(self.handle as *mut std::ffi::c_void);
        }
    }
}

/// Native code in executable memory that this process didn't map itself:
/// the code of one [`CachedUnit`], for [`CodeMemory::linked`].
///
/// [`CodeMemory::linked`]: crate::CodeMemory::linked
#[derive(Debug, Clone)]
pub struct LinkedCode {
    pub(crate) base: usize,
    pub(crate) len: usize,
    /// The shared object containing the code, or `None` for the program.
    pub(crate) owner: Option<Arc<SharedObject>>,
}

impl LinkedCode {
    /// Size of the code in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: records with empty code are skipped.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The code of one execute segment from an earlier run, with the metadata
/// `libriscv` saved along with it (copied out of the object).
#[derive(Debug, Clone)]
pub struct CachedUnit {
    pub meta: Vec<u8>,
    pub code: LinkedCode,
}

/// Read the record at `record` (`RECORD_WORDS` readable words).
///
/// # Safety
///
/// `record` must point at `RECORD_WORDS` readable, aligned `u64`s, and if
/// they are a record (the magic matches), its code and metadata must be
/// where it says, in memory that stays mapped as long as `owner`, or for
/// the program's lifetime if that's `None`.
#[allow(unsafe_code)]
unsafe fn read_record(record: *const u64, owner: Option<&Arc<SharedObject>>) -> Option<CachedUnit> {
    // SAFETY: the caller guarantees `record` points at a whole record.
    let words = unsafe { std::ptr::read(record.cast::<[u64; RECORD_WORDS]>()) };
    let [magic, code_rel, code_len, meta_rel, meta_len, _] = words;
    if magic != RECORD_MAGIC
        || code_len == 0
        || code_len > MAX_LEN
        || meta_len == 0
        || meta_len > MAX_LEN
    {
        return None;
    }
    let base = record as usize;
    let code = (base + 8).wrapping_add(code_rel as usize);
    let meta = (base + 24).wrapping_add(meta_rel as usize);
    // SAFETY: a record's metadata is `meta_len` bytes at `meta`, mapped
    // (read-only data) as long as `owner` (caller's guarantee); it is
    // copied out right away.
    let meta = unsafe { std::slice::from_raw_parts(meta as *const u8, meta_len as usize) }.to_vec();
    Some(CachedUnit {
        meta,
        code: LinkedCode {
            base: code,
            len: code_len as usize,
            owner: owner.cloned(),
        },
    })
}

/// The cached code linked into this program: the records of every object
/// that was linked into it.
pub fn embedded_units() -> Vec<CachedUnit> {
    #[cfg(target_os = "linux")]
    return embedded::units();
    #[cfg(not(target_os = "linux"))]
    Vec::new()
}

/// Load the shared object at `path` (`dlopen`) and return its records. The
/// object stays loaded as long as code from it is in use.
///
/// Loading a shared object runs its code (its initializers, and later the
/// code of its records): only load files you would also run.
pub fn load_shared_object(path: &Path) -> io::Result<Vec<CachedUnit>> {
    #[cfg(target_os = "linux")]
    return sys::load(path);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "loading cached JIT code is only supported on Linux",
        ))
    }
}

#[cfg(target_os = "linux")]
mod embedded {
    use super::{CachedUnit, RECORD_WORDS, read_record};

    /// An empty record, so that the section (and with it the `__start_` and
    /// `__stop_` symbols) exists in every program, also without any cached
    /// code linked in. Its magic is 0: it's skipped. Not `mut`, so that the
    /// section's flags (read-only data) match those of the records.
    #[allow(unsafe_code)]
    // SAFETY: `link_section` places a plain array of integers in our
    // section. Nothing relies on its position besides `units`, which reads
    // the section as an array of records, whatever their order.
    #[unsafe(link_section = "libriscv_jitcache")]
    #[used]
    static EMPTY: [u64; RECORD_WORDS] = [0; RECORD_WORDS];

    #[allow(unsafe_code)]
    // SAFETY: the linker defines these for the section `libriscv_jitcache`
    // (whose name is a C identifier), which `EMPTY` guarantees exists: its
    // start and end. Only their addresses are used.
    unsafe extern "C" {
        static __start_libriscv_jitcache: u64;
        static __stop_libriscv_jitcache: u64;
    }

    pub(super) fn units() -> Vec<CachedUnit> {
        // Keep `EMPTY` in the program whenever this function is.
        std::hint::black_box(&EMPTY);
        let start = (&raw const __start_libriscv_jitcache) as usize;
        let stop = (&raw const __stop_libriscv_jitcache) as usize;
        let size = RECORD_WORDS * 8;
        let mut units = Vec::new();
        let mut at = start;
        while at + size <= stop {
            #[allow(unsafe_code)]
            // SAFETY: the section is an array of records (`EMPTY` and those
            // of the linked objects, all `RECORD_WORDS` aligned words with
            // no padding between them), mapped for the program's lifetime,
            // and `at` is a record inside it. A record's code and metadata
            // are in the same program (its relocations point there).
            if let Some(unit) = unsafe { read_record(at as *const u64, None) } {
                units.push(unit);
            }
            at += size;
        }
        units
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::io;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::sync::Arc;

    use super::{CachedUnit, RECORD_WORDS, SharedObject, TABLE_MAGIC, TABLE_SYMBOL, TABLE_WORDS};

    const RTLD_NOW: c_int = 2;
    const RTLD_LOCAL: c_int = 0;

    #[allow(unsafe_code)]
    // SAFETY: the POSIX declarations of these libc functions.
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub(super) fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *mut c_char;
    }

    /// The pending `dlerror` message.
    #[allow(unsafe_code)]
    fn error(what: &str) -> io::Error {
        // SAFETY: no preconditions; returns null or a NUL-terminated string
        // valid until the next dl* call on this thread.
        let e = unsafe { dlerror() };
        let message = if e.is_null() {
            String::from("unknown error")
        } else {
            // SAFETY: `e` is the non-null string just returned, copied
            // before any other dl* call.
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        };
        io::Error::other(format!("{what}: {message}"))
    }

    pub(super) fn load(path: &Path) -> io::Result<Vec<CachedUnit>> {
        // `dlopen` searches the library path for names without a slash.
        let path = if path.components().count() == 1 {
            Path::new(".").join(path)
        } else {
            path.to_path_buf()
        };
        let name = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in path"))?;
        #[allow(unsafe_code)]
        // SAFETY: `name` is a NUL-terminated path. Loading runs the
        // object's initializers: the caller trusts the file (see
        // `load_shared_object`); the ones `libriscv` writes have none.
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            return Err(error("dlopen"));
        }
        // Closed again when the last unit's code is dropped (or below).
        let owner = Arc::new(SharedObject {
            handle: handle as usize,
        });
        let symbol = CString::new(TABLE_SYMBOL).map_err(|_| error("symbol"))?;
        #[allow(unsafe_code)]
        // SAFETY: `handle` is a live `dlopen` handle, `symbol` NUL-terminated.
        let table = unsafe { dlsym(handle, symbol.as_ptr()) };
        if table.is_null() {
            return Err(error("no cached JIT code in the shared object"));
        }
        #[allow(unsafe_code)]
        // SAFETY: the table symbol is `TABLE_WORDS` aligned words (the
        // format this crate's documentation describes), mapped while the
        // object is loaded.
        let [magic, count, rel] = unsafe { std::ptr::read(table.cast::<[u64; TABLE_WORDS]>()) };
        if magic != TABLE_MAGIC || count > 1 << 20 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid cached JIT code table",
            ));
        }
        let records = (table as usize + 16).wrapping_add(rel as usize);
        let mut units = Vec::new();
        for i in 0..count as usize {
            let record = records + i * RECORD_WORDS * 8;
            #[allow(unsafe_code)]
            // SAFETY: the table refers to `count` consecutive records in
            // the same object, whose code and metadata are in it too; it
            // stays loaded as long as `owner`, which every unit holds.
            if let Some(unit) = unsafe { super::read_record(record as *const u64, Some(&owner)) } {
                units.push(unit);
            }
        }
        Ok(units)
    }
}
