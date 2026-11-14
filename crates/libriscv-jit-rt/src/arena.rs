//! Guest memory arenas whose low pages can be a shared, sealed image: the
//! memory side of C++ `shared_rodata.cpp` (`SharedRodataImage`), for
//! `libriscv`'s `use_shared_rodata` option (with its `jit` feature).
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

use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// True when [`RodataImage`]s can be created on this platform (sealable
/// anonymous files, ie. Linux memfds).
pub const fn shared_rodata_supported() -> bool {
    cfg!(target_os = "linux")
}

/// The host page size, which is what an image's length must be a multiple
/// of to be attached.
pub fn host_page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf has no preconditions.
        #[allow(unsafe_code)]
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(size)
            .ok()
            .filter(|&s| s > 0)
            .unwrap_or(4096)
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

/// Zero-initialized guest memory: an anonymous private mapping of `len`
/// bytes (`MAP_NORESERVE`, so untouched pages cost nothing), owned
/// exclusively like a `Box<[u8]>`. The low part of it can be replaced by a
/// shared image ([`GuestArena::attach`]).
pub struct GuestArena {
    ptr: NonNull<u8>,
    len: usize,
    /// Whether `ptr` is a mapping of ours (false for the empty arena).
    mapped: bool,
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
            mapped: false,
        }
    }

    /// `len` bytes of zeroes.
    #[cfg(unix)]
    pub fn new(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Ok(Self::empty());
        }
        // SAFETY: a fresh anonymous mapping at an address of the kernel's
        // choosing affects no existing memory.
        #[allow(unsafe_code)]
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let ptr =
            NonNull::new(addr.cast::<u8>()).ok_or_else(|| io::Error::other("null mapping"))?;
        Ok(Self {
            ptr,
            len,
            mapped: true,
        })
    }

    /// Replace the first `image.len()` bytes of the arena with `image`,
    /// mapped copy-on-write: they read as the image, and a store to one of
    /// its pages makes a private copy of that page.
    ///
    /// The image must fit in the arena and be a whole number of host pages
    /// long. On failure, the arena is unchanged, unless the mapping
    /// operation itself failed after the kernel had already discarded the
    /// old pages: then the range is mapped again with fresh zeroes, and the
    /// process aborts if even that fails (it couldn't access its own
    /// memory any more).
    #[cfg(target_os = "linux")]
    pub fn attach(&mut self, image: &RodataImage) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        let len = image.len();
        if len == 0 || len > self.len || !len.is_multiple_of(host_page_size()) || !self.mapped {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let base = self.ptr.as_ptr().cast::<libc::c_void>();
        // SAFETY: `[base, base + len)` is inside our own mapping (checked
        // above) and `&mut self` means nothing borrows it. MAP_FIXED
        // replaces those pages with the image's: `len` is the whole sealed
        // file, a multiple of the page size, so every page is backed and
        // can neither change nor be truncated away.
        #[allow(unsafe_code)]
        let addr = unsafe {
            libc::mmap(
                base,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_FIXED,
                image.file.as_raw_fd(),
                0,
            )
        };
        if addr == base {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        // The old pages may be gone: map zeroes over them again.
        // SAFETY: as above, our own range, borrowed by nothing.
        #[allow(unsafe_code)]
        let again = unsafe {
            libc::mmap(
                base,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if again != base {
            std::process::abort();
        }
        Err(error)
    }

    /// Images can't be created on this platform.
    #[cfg(not(target_os = "linux"))]
    pub fn attach(&mut self, _image: &RodataImage) -> io::Result<()> {
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

#[cfg(not(unix))]
impl GuestArena {
    /// `len` bytes of zeroes.
    pub fn new(len: usize) -> io::Result<Self> {
        let _ = len;
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

impl Drop for GuestArena {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.mapped {
            // SAFETY: our own mapping (attached images included, which are
            // inside it), unmapped once; `&mut self` means nothing borrows it.
            #[allow(unsafe_code)]
            unsafe {
                libc::munmap(self.ptr.as_ptr().cast(), self.len);
            }
        }
    }
}

impl Deref for GuestArena {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is `len` readable bytes for as long as `self` lives
        // (or dangling and aligned with `len` 0), owned by `self`, and only
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
/// `SharedRodataImage`): a memfd named `libriscv-rodata`, plus a read-only
/// view of it for comparing contents.
pub struct RodataImage {
    file: File,
    view: memmap2::Mmap,
}

impl std::fmt::Debug for RodataImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RodataImage")
            .field("len", &self.view.len())
            .finish()
    }
}

impl RodataImage {
    /// A sealed image holding `bytes`, which must not be empty.
    #[cfg(target_os = "linux")]
    pub fn new(bytes: &[u8]) -> io::Result<Self> {
        use std::io::Write;
        use std::os::fd::FromRawFd;
        if bytes.is_empty() {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        // SAFETY: the name is a valid C string; no memory is affected.
        #[allow(unsafe_code)]
        let fd = unsafe {
            libc::memfd_create(
                c"libriscv-rodata".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `fd` is a freshly created descriptor that nothing else
        // owns.
        #[allow(unsafe_code)]
        let mut file = unsafe { File::from_raw_fd(fd) };
        // Filled with write(), so no writable mapping ever exists, which
        // F_SEAL_WRITE requires.
        file.write_all(bytes)?;
        // SAFETY: fcntl on our own descriptor, affecting no memory.
        #[allow(unsafe_code)]
        let sealed = unsafe {
            libc::fcntl(
                fd,
                libc::F_ADD_SEALS,
                libc::F_SEAL_SEAL | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_WRITE,
            )
        };
        if sealed != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the file is sealed: its contents and length can't change
        // for as long as the view lives.
        #[allow(unsafe_code)]
        let view = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { file, view })
    }

    /// Images can't be created on this platform.
    #[cfg(not(target_os = "linux"))]
    pub fn new(_bytes: &[u8]) -> io::Result<Self> {
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }

    /// The image's contents.
    pub fn bytes(&self) -> &[u8] {
        &self.view
    }

    /// The image's length in bytes.
    pub fn len(&self) -> usize {
        self.view.len()
    }

    /// True for an empty image (never the case for a created one).
    pub fn is_empty(&self) -> bool {
        self.view.is_empty()
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn attached_image_reads_through_and_copies_on_write() {
        let page = host_page_size();
        let mut content = vec![0u8; 2 * page];
        content[0] = 0xAA;
        content[page + 1] = 0xBB;
        let image = RodataImage::new(&content).unwrap();
        assert_eq!(image.bytes(), &content[..]);

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
        assert_eq!(image.bytes()[1], 0);
    }

    #[test]
    fn attach_refuses_partial_pages_and_oversized_images() {
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
        use std::io::Write;
        let page = host_page_size();
        let image = RodataImage::new(&vec![3u8; page]).unwrap();
        assert!((&image.file).write_all(b"x").is_err());
        assert!(image.file.set_len(0).is_err());
        assert!(image.file.set_len(2 * page as u64).is_err());
    }
}
