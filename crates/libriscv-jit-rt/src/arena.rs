//! Port of the memory side of `shared_rodata.cpp` (`SharedRodataImage`, the
//! memfd and the mapping of `create_shared_rodata_image`), for `libriscv`'s
//! `use_shared_rodata` option (with its `jit` feature).
//!
//! A [`GuestArena`] is one anonymous private mapping that `libriscv` uses
//! as its flat arena, exactly like the `Vec<u8>` it replaces: a contiguous
//! `[u8]` that the interpreter indexes and native code addresses from its
//! base. [`GuestArena::attach`] maps a [`RodataImage`] over the start of it,
//! so that machines loaded from the same program hold its text and rodata
//! in host memory once.
//!
//! # Differences from C++
//!
//! C++ maps the image `PROT_READ | MAP_SHARED`, relying on its arena's
//! write boundary to keep stores away, and turns sharing off whenever
//! something could store there anyway (virtual paging, N-bit arenas). Here
//! the image is mapped `PROT_READ | PROT_WRITE | MAP_PRIVATE`: the pages are
//! the memfd's page cache pages, shared between every machine, until a
//! store reaches one, which then gets a private copy from the kernel
//! (copy-on-write). So the arena stays an ordinary writable `[u8]`, a store
//! that does get there (a page the guest made writable with `mprotect`, an
//! N-bit arena, the host) behaves as without sharing, and nothing in the
//! access paths changes. The image file is sealed against writes, growing
//! and shrinking, so the bytes behind those pages never change and never
//! disappear (no `SIGBUS`).
//!
//! The arena itself is a `memmap2` anonymous mapping on every platform.
//! Images need sealable memfds and a fixed-address `mmap`, which `memmap2`
//! doesn't offer: the few Linux calls for those are declared in [`sys`] by
//! hand (like `dlopen` in `linked.rs`), for the architectures whose `mmap`
//! flag values they spell out.

use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use memmap2::{Mmap, MmapMut, MmapOptions};

/// True when [`RodataImage`]s can be created on this platform (sealable
/// anonymous files, ie. Linux memfds).
pub const fn shared_rodata_supported() -> bool {
    sys::SUPPORTED
}

/// The host page size, which is what an image's length must be a multiple
/// of to be attached (4096 where images aren't supported).
pub fn host_page_size() -> usize {
    sys::page_size()
}

/// Zero-initialized guest memory: an anonymous private mapping of `len`
/// bytes (`MAP_NORESERVE` where the platform has it, so untouched pages
/// cost nothing), owned exclusively like a `Box<[u8]>`. The low part of it
/// can be replaced by a shared image ([`GuestArena::attach`]).
pub struct GuestArena {
    /// The mapping's base and length, cached so that `deref` is two loads
    /// (the arena is on every interpreter memory access).
    ptr: NonNull<u8>,
    len: usize,
    /// The mapping `ptr` points into (none for the empty arena), unmapped
    /// on drop, attached images included.
    map: Option<MmapMut>,
}

// SAFETY: the arena exclusively owns its mapping, like a `Box<[u8]>`, and
// only hands it out through `&self`/`&mut self` borrows.
#[allow(unsafe_code)]
unsafe impl Send for GuestArena {}
// SAFETY: as for `Send`: shared access is only ever to `&[u8]`.
#[allow(unsafe_code)]
unsafe impl Sync for GuestArena {}

impl std::fmt::Debug for GuestArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestArena")
            .field("base", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl GuestArena {
    /// An arena of no bytes, without a mapping.
    pub const fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            map: None,
        }
    }

    /// `len` bytes of zeroes.
    pub fn new(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Ok(Self::empty());
        }
        let mut map = MmapOptions::new().len(len).no_reserve_swap().map_anon()?;
        let ptr = NonNull::new(map.as_mut_ptr()).ok_or_else(|| io::Error::other("null mapping"))?;
        Ok(Self {
            ptr,
            len: map.len(),
            map: Some(map),
        })
    }

    /// Replace the first `image.len()` bytes of the arena with `image`,
    /// mapped copy-on-write: they read as the image, and a store to one of
    /// its pages makes a private copy of that page.
    ///
    /// The image must fit in the arena and be a whole number of host pages
    /// long. On failure the range reads as the image's bytes anyway, as
    /// private anonymous memory: callers attach only an image that holds
    /// what the range holds already (just published from it) or is about
    /// to (verified against the program), so either way the guest sees the
    /// same bytes. Every step that can fail for want of resources (mapping
    /// the view to restore from) comes before the arena is touched; the
    /// process aborts only if re-mapping the arena's own range fails, ie.
    /// it couldn't access its own memory any more.
    pub fn attach(&mut self, image: &RodataImage) -> io::Result<()> {
        let len = image.len();
        if !sys::SUPPORTED
            || len == 0
            || len > self.len
            || !len.is_multiple_of(host_page_size())
            || self.map.is_none()
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        // The bytes to restore if the mapping fails (and the kernel had
        // already dropped the old pages).
        let view = image.view()?;
        let base = self.ptr.as_ptr();
        // SAFETY: `[base, base + len)` is inside our own mapping (checked
        // above) and `&mut self` means nothing borrows it. The image is
        // the whole sealed file, a multiple of the page size, so every
        // page is backed and can neither change nor be truncated away.
        #[allow(unsafe_code)]
        let mapped = unsafe { sys::map_image_fixed(base, len, &image.file) };
        let Err(error) = mapped else {
            return Ok(());
        };
        // SAFETY: as above, our own range, borrowed by nothing.
        #[allow(unsafe_code)]
        let zeroed = unsafe { sys::map_zeroes_fixed(base, len) };
        if zeroed.is_err() {
            // Part of the arena may be unmapped: `deref` would be unsound.
            std::process::abort();
        }
        if let Some(range) = self.get_mut(..len) {
            range.copy_from_slice(&view);
        }
        Err(error)
    }
}

impl Deref for GuestArena {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is `len` readable bytes owned by `map` for as long
        // as `self` lives (or dangling and aligned with `len` 0), and only
        // mutated through `&mut self`.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl DerefMut for GuestArena {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as for `deref`; every page is writable, attached image
        // pages included (copy-on-write), and `&mut self` is exclusive.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

/// A sealed, immutable image of a program's read-only memory (C++
/// `SharedRodataImage`): a memfd named `libriscv-rodata`. Its contents are
/// read through a temporary [`RodataImage::view`], so that a live image
/// costs no mapping besides the arenas it is attached to.
pub struct RodataImage {
    file: File,
    len: usize,
}

impl std::fmt::Debug for RodataImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RodataImage")
            .field("len", &self.len)
            .finish()
    }
}

impl RodataImage {
    /// A sealed image holding `bytes`, which must not be empty.
    pub fn new(bytes: &[u8]) -> io::Result<Self> {
        use std::io::Write;
        if bytes.is_empty() {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let mut file = sys::new_memfd()?;
        // Filled with write(), so no writable mapping ever exists, which
        // F_SEAL_WRITE requires.
        file.write_all(bytes)?;
        sys::seal(&file)?;
        Ok(Self {
            file,
            len: bytes.len(),
        })
    }

    /// A read-only mapping of the image's contents (C++ `inspect`), for as
    /// long as the caller keeps it.
    pub fn view(&self) -> io::Result<impl Deref<Target = [u8]> + use<>> {
        // SAFETY: the file is sealed: its contents and length can't change
        // for as long as the view lives.
        #[allow(unsafe_code)]
        let view = unsafe { Mmap::map(&self.file)? };
        if view.len() != self.len {
            return Err(io::Error::other("image changed size"));
        }
        Ok(view)
    }

    /// The image's length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True for an empty image (never the case for a created one).
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The Linux calls `memmap2` doesn't cover, declared by hand with the
/// generic flag values of the architectures listed (mips, powerpc and sparc
/// spell `MAP_ANONYMOUS`/`MAP_NORESERVE` differently, so they go without).
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )
))]
mod sys {
    use std::ffi::{c_char, c_int, c_long, c_uint, c_void};
    use std::fs::File;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd};

    pub(super) const SUPPORTED: bool = true;

    const PROT_READ: c_int = 1;
    const PROT_WRITE: c_int = 2;
    const MAP_PRIVATE: c_int = 0x02;
    const MAP_FIXED: c_int = 0x10;
    const MAP_ANONYMOUS: c_int = 0x20;
    const MAP_NORESERVE: c_int = 0x4000;
    const MFD_CLOEXEC: c_uint = 1;
    const MFD_ALLOW_SEALING: c_uint = 2;
    const F_ADD_SEALS: c_int = 1033;
    const F_SEAL_SEAL: c_int = 1;
    const F_SEAL_SHRINK: c_int = 2;
    const F_SEAL_GROW: c_int = 4;
    const F_SEAL_WRITE: c_int = 8;
    /// glibc and musl agree on Linux.
    const SC_PAGESIZE: c_int = 30;

    // SAFETY: the Linux declarations of these libc functions (`off_t` is
    // 64-bit on the architectures above).
    #[allow(unsafe_code)]
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
        fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
        safe fn sysconf(name: c_int) -> c_long;
    }

    pub(super) fn page_size() -> usize {
        usize::try_from(sysconf(SC_PAGESIZE))
            .ok()
            .filter(|&s| s > 0)
            .unwrap_or(4096)
    }

    /// `mmap(MAP_FIXED)` over `[base, base + len)`; Ok when it landed there.
    ///
    /// # Safety
    ///
    /// The range must be a mapping the caller owns and nothing borrows:
    /// its pages are replaced.
    #[allow(unsafe_code)]
    unsafe fn map_fixed(base: *mut u8, len: usize, flags: c_int, fd: c_int) -> io::Result<()> {
        // SAFETY: the caller's contract; the result is checked.
        let addr = unsafe {
            mmap(
                base.cast(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE | flags,
                fd,
                0,
            )
        };
        if addr == base.cast() {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Map `file` copy-on-write over `[base, base + len)`.
    ///
    /// # Safety
    ///
    /// As for `map_fixed`, and `file` must be at least `len` bytes long
    /// for as long as the mapping lives.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_image_fixed(base: *mut u8, len: usize, file: &File) -> io::Result<()> {
        #[cfg(test)]
        if FAIL_AFTER_UNMAP.take() {
            // What a failing kernel may leave behind: the old pages gone.
            // SAFETY: the caller's contract.
            unsafe { map_zeroes_fixed(base, len)? };
            return Err(io::Error::from(io::ErrorKind::OutOfMemory));
        }
        // SAFETY: the caller's contract.
        unsafe { map_fixed(base, len, 0, file.as_raw_fd()) }
    }

    #[cfg(test)]
    thread_local! {
        /// Make the next `map_image_fixed` on this thread fail after
        /// dropping the range's pages.
        pub(super) static FAIL_AFTER_UNMAP: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    /// Map fresh zeroes over `[base, base + len)`.
    ///
    /// # Safety
    ///
    /// As for `map_fixed`.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_zeroes_fixed(base: *mut u8, len: usize) -> io::Result<()> {
        // SAFETY: the caller's contract.
        unsafe { map_fixed(base, len, MAP_ANONYMOUS, -1) }
    }

    /// A new empty sealable memfd named `libriscv-rodata`.
    pub(super) fn new_memfd() -> io::Result<File> {
        // SAFETY: the name is a valid C string; no memory is affected.
        #[allow(unsafe_code)]
        let fd =
            unsafe { memfd_create(c"libriscv-rodata".as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `fd` is a freshly created descriptor that nothing else
        // owns.
        #[allow(unsafe_code)]
        let file = unsafe { File::from_raw_fd(fd) };
        Ok(file)
    }

    /// Seal `file` against writes, growing, shrinking and further seals.
    pub(super) fn seal(file: &File) -> io::Result<()> {
        let seals = F_SEAL_SEAL | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE;
        // SAFETY: fcntl on a descriptor we own, affecting no memory.
        #[allow(unsafe_code)]
        let sealed = unsafe { fcntl(file.as_raw_fd(), F_ADD_SEALS, seals) };
        if sealed == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

/// Images aren't supported here: arenas still work (they are `memmap2`
/// mappings), nothing is ever attached.
#[cfg(not(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )
)))]
mod sys {
    use std::fs::File;
    use std::io;

    pub(super) const SUPPORTED: bool = false;

    pub(super) const fn page_size() -> usize {
        4096
    }

    fn unsupported<T>() -> io::Result<T> {
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }

    /// Never called: `attach` checks `SUPPORTED` first.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_image_fixed(_: *mut u8, _: usize, _: &File) -> io::Result<()> {
        unsupported()
    }

    /// Never called: `attach` checks `SUPPORTED` first.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_zeroes_fixed(_: *mut u8, _: usize) -> io::Result<()> {
        unsupported()
    }

    pub(super) fn new_memfd() -> io::Result<File> {
        unsupported()
    }

    pub(super) fn seal(_: &File) -> io::Result<()> {
        unsupported()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn arena_is_zeroed_and_writable() {
        let mut arena = GuestArena::new(3 * 4096 + 5).unwrap();
        assert_eq!(arena.len(), 3 * 4096 + 5);
        assert!(arena.iter().all(|&b| b == 0));
        arena[3 * 4096 + 4] = 9;
        assert_eq!(arena[3 * 4096 + 4], 9);
        assert!(GuestArena::new(0).unwrap().is_empty());
        assert!(GuestArena::empty().is_empty());
    }

    #[test]
    fn attached_image_reads_through_and_copies_on_write() {
        if !shared_rodata_supported() {
            return;
        }
        let page = host_page_size();
        let mut content = vec![0u8; 2 * page];
        content[0] = 0xAA;
        content[page + 1] = 0xBB;
        let image = RodataImage::new(&content).unwrap();
        assert_eq!(&*image.view().unwrap(), &content[..]);

        let mut a = GuestArena::new(4 * page).unwrap();
        let mut b = GuestArena::new(4 * page).unwrap();
        a[3 * page] = 7;
        a.attach(&image).unwrap();
        b.attach(&image).unwrap();
        assert_eq!(&a[..2 * page], &content[..]);
        assert_eq!(&b[..2 * page], &content[..]);
        // Above the image the arena is as it was.
        assert_eq!(a[3 * page], 7);

        // A store is private to the arena that made it.
        a[1] = 0xCC;
        assert_eq!(a[1], 0xCC);
        assert_eq!(b[1], 0);
        assert_eq!(image.view().unwrap()[1], 0);
    }

    #[test]
    fn attach_refuses_partial_pages_and_oversized_images() {
        if !shared_rodata_supported() {
            return;
        }
        let page = host_page_size();
        let image = RodataImage::new(&vec![1u8; page + 1]).unwrap();
        let mut arena = GuestArena::new(4 * page).unwrap();
        assert!(arena.attach(&image).is_err());
        assert!(arena.iter().all(|&b| b == 0));

        let image = RodataImage::new(&vec![1u8; 8 * page]).unwrap();
        assert!(arena.attach(&image).is_err());
        assert!(GuestArena::empty().attach(&image).is_err());
    }

    #[test]
    fn image_is_sealed() {
        if !shared_rodata_supported() {
            return;
        }
        use std::io::Write;
        let page = host_page_size();
        let image = RodataImage::new(&vec![3u8; page]).unwrap();
        assert!((&image.file).write_all(b"x").is_err());
        assert!(image.file.set_len(0).is_err());
        assert!(image.file.set_len(2 * page as u64).is_err());
    }

    /// A mapping that fails after the kernel dropped the old pages leaves
    /// the image's bytes in place, privately.
    #[cfg(all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    ))]
    #[test]
    fn failed_attach_restores_the_image_bytes() {
        let page = host_page_size();
        let content: Vec<u8> = (0..2 * page).map(|i| (i % 251) as u8).collect();
        let image = RodataImage::new(&content).unwrap();
        let mut arena = GuestArena::new(4 * page).unwrap();
        arena[..2 * page].copy_from_slice(&content);
        arena[3 * page] = 7;
        sys::FAIL_AFTER_UNMAP.set(true);
        assert!(arena.attach(&image).is_err());
        assert_eq!(&arena[..2 * page], &content[..]);
        assert_eq!(arena[3 * page], 7);
        // Private memory: stores stay in this arena.
        arena[0] ^= 0xFF;
        assert_eq!(image.view().unwrap()[0], content[0]);
    }

    #[test]
    fn images_are_unsupported_elsewhere() {
        if shared_rodata_supported() {
            return;
        }
        assert!(RodataImage::new(&[1]).is_err());
    }
}
