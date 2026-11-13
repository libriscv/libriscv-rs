//! Port of the `dlopen`/`dlsym` loading of translations in `tr_compiler.cpp`
//! and `tr_translate.cpp`.
//!
//! Loading `libriscv`'s ahead-of-time translations from shared objects.
//!
//! `libriscv`'s `aot` module translates execute segments to safe Rust and
//! compiles the translation into a `cdylib` that exports two functions:
//!
//! * [`META_SYMBOL`] `extern "C" fn() -> RawMeta`: the translation's
//!   metadata, `u64` words that `libriscv` parses (its keys and entries);
//! * [`ENTER_SYMBOL`] `unsafe extern "C" fn(*mut RawCall) -> u8`: run the
//!   translated code of a segment from a decoder entry, with the guest's
//!   registers and arena (see [`AotLibrary::enter`]).
//!
//! Unlike JIT code, the translated code itself is safe Rust: it can only
//! touch the registers and the arena it is given, through bounds-checked
//! slices. What can't be checked is that the shared object really is such
//! a translation: like any native library, loading it runs its code.

use std::ffi::c_void;
use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::linked::SharedObject;

/// The exported metadata function.
pub const META_SYMBOL: &str = "libriscv_aot_meta";
/// The exported entry function.
pub const ENTER_SYMBOL: &str = "libriscv_aot_enter";

/// The arguments of the entry function. Mirrored by the prologue `libriscv`
/// writes into every shared object.
#[repr(C)]
#[derive(Debug)]
pub struct RawCall {
    pub seg: u64,
    pub index: u64,
    pub regs: *mut u64,
    pub fregs: *mut u64,
    pub arena: *mut u8,
    pub arena_len: u64,
    /// `[pc, counter, max, frm]` in, `[exit_kind, pc, counter, 0]` out.
    pub io: [u64; 4],
}

/// The result of the metadata function.
#[repr(C)]
#[derive(Debug)]
pub struct RawMeta {
    pub words: *const u64,
    pub len: u64,
}

type MetaFn = extern "C" fn() -> RawMeta;
type EnterFn = unsafe extern "C" fn(*mut RawCall) -> u8;

/// Refuse metadata beyond this many words: no real translation comes close.
const MAX_META: u64 = 1 << 32;

/// A loaded translation.
#[derive(Debug)]
pub struct AotLibrary {
    meta: Vec<u64>,
    enter: EnterFn,
    /// Keeps the shared object loaded while `enter` may be called.
    _owner: Arc<SharedObject>,
}

impl AotLibrary {
    /// Load the shared object at `path` (`dlopen`) and read its metadata.
    /// The object stays loaded as long as the returned library.
    ///
    /// Loading a shared object runs its code (its initializers, and later
    /// its translated code): only load files you would also run.
    pub fn load(path: &Path) -> io::Result<Arc<Self>> {
        #[cfg(target_os = "linux")]
        {
            use crate::linked::sys;
            let owner = sys::open(path)?;
            let what = "no libriscv translation in the shared object";
            let meta = sys::symbol(&owner, META_SYMBOL, what)?;
            let enter = sys::symbol(&owner, ENTER_SYMBOL, what)?;
            #[allow(unsafe_code)]
            // SAFETY: the symbol is the metadata function of the prologue
            // `libriscv` writes, with exactly this signature. (A shared
            // object is trusted, see above.)
            let meta = unsafe { std::mem::transmute::<*mut c_void, MetaFn>(meta) };
            #[allow(unsafe_code)]
            // SAFETY: the symbol is the entry function of that prologue,
            // with exactly this signature.
            let enter = unsafe { std::mem::transmute::<*mut c_void, EnterFn>(enter) };
            let RawMeta { words, len } = meta();
            if words.is_null() || len > MAX_META || !words.is_aligned() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid libriscv translation metadata",
                ));
            }
            #[allow(unsafe_code)]
            // SAFETY: the function returns its `static META: &[u64]`, which
            // lives in the object (mapped while `owner` is alive), aligned
            // and non-null (checked); copied right away.
            let meta = unsafe { std::slice::from_raw_parts(words, len as usize) }.to_vec();
            Ok(Arc::new(Self {
                meta,
                enter,
                _owner: owner,
            }))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = path;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "loading translations is only supported on Linux",
            ))
        }
    }

    /// The translation's metadata.
    pub fn meta(&self) -> &[u64] {
        &self.meta
    }

    /// Run the translated code of segment `seg` from decoder entry `index`
    /// (see `libriscv::aot::AotEnter` for `io`). Returns false if the code
    /// couldn't be entered there.
    pub fn enter(
        &self,
        seg: usize,
        index: usize,
        regs: &mut [u64; 32],
        fregs: &mut [u64; 32],
        arena: &mut [u8],
        io: &mut [u64; 4],
    ) -> bool {
        let mut call = RawCall {
            seg: seg as u64,
            index: index as u64,
            regs: regs.as_mut_ptr(),
            fregs: fregs.as_mut_ptr(),
            arena: arena.as_mut_ptr(),
            arena_len: arena.len() as u64,
            io: *io,
        };
        #[allow(unsafe_code)]
        // SAFETY: the entry function's contract: `call` is valid, its
        // pointers come from the borrows above (32 registers each, and
        // `arena_len` bytes), which outlive the call and aren't used
        // otherwise until it returns. The translated code behind it is safe
        // Rust, which can't access anything else.
        let entered = unsafe { (self.enter)(&raw mut call) } != 0;
        *io = call.io;
        entered
    }
}
