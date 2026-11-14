//! Port of `posix/filedesc.hpp`: the sandboxed file descriptor table.
//!
//! The guest sees virtual file descriptors. 0, 1 and 2 are the standard
//! streams, files opened by the guest get numbers from `FILE_D_BASE` up.
//! C++ maps them to raw host file descriptors and calls the host's POSIX
//! functions on them; here they own `std::fs::File`s, so there is no raw
//! file descriptor (and no `unsafe`) anywhere. What a guest can reach is
//! decided by the system call handlers (`permit_filesystem`, `proxy_mode`)
//! and the filters, which are closures instead of `std::function`s taking
//! the userdata pointer.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;

/// Decides whether a path may be opened, and may rewrite it
/// (C++ `filter_open` / `filter_readlink`).
pub type PathFilter = Box<dyn FnMut(&mut String) -> bool + Send + Sync>;
/// Decides whether a path may be inspected (C++ `filter_stat`).
pub type StatFilter = Box<dyn FnMut(&str) -> bool + Send + Sync>;
/// Decides whether an ioctl request is allowed (C++ `filter_ioctl`).
pub type IoctlFilter = Box<dyn FnMut(u64) -> bool + Send + Sync>;

/// An open guest file.
#[derive(Debug)]
pub struct OpenFile {
    pub file: File,
    /// The (filtered) path it was opened with; relative `openat` lookups
    /// against a directory descriptor use it.
    pub path: String,
    /// The access mode bits (`O_ACCMODE`) it was opened with, for `F_GETFL`.
    pub access: i32,
}

/// What a virtual file descriptor refers to (C++ `translate`).
#[derive(Debug)]
pub enum Target<'a> {
    /// Host standard input, output or error.
    Stdio(i32),
    File(&'a mut OpenFile),
}

/// The file descriptor table and sandbox policy (C++ `FileDescriptors`).
pub struct FileDescriptors {
    files: BTreeMap<i32, OpenFile>,
    /// The working directory the guest sees (a fake root by default).
    pub cwd: String,
    file_counter: i32,
    /// Allow opening files at all (subject to `filter_open`).
    pub permit_filesystem: bool,
    /// Allow sockets (not ported yet; kept for API parity).
    pub permit_sockets: bool,
    /// Proxy mode: the guest may write to files and see host details
    /// (exact time). Disables most of the sandbox.
    pub proxy_mode: bool,
    pub filter_open: Option<PathFilter>,
    pub filter_readlink: Option<PathFilter>,
    pub filter_stat: Option<StatFilter>,
    pub filter_ioctl: Option<IoctlFilter>,
    /// Files whose `mmap` without an address (`ld.so` mapping a shared
    /// library) goes to a fixed guest address: (filtered path, address).
    /// The address is used when it is page aligned, the whole mapping fits
    /// in the arena and that range of the mmap area is free; otherwise the
    /// mapping goes where it would have gone. Not in C++: it keeps libraries
    /// at the same address in every program, so code saved for them (the
    /// JIT cache, AOT translations, both keyed by address) is found again.
    pub pinned_mappings: Vec<(String, u64)>,
}

impl core::fmt::Debug for FileDescriptors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FileDescriptors")
            .field("files", &self.files)
            .field("cwd", &self.cwd)
            .field("permit_filesystem", &self.permit_filesystem)
            .field("permit_sockets", &self.permit_sockets)
            .field("proxy_mode", &self.proxy_mode)
            .finish_non_exhaustive()
    }
}

impl Default for FileDescriptors {
    fn default() -> Self {
        Self {
            files: BTreeMap::new(),
            cwd: "/home".to_owned(),
            file_counter: Self::FILE_D_BASE,
            permit_filesystem: false,
            permit_sockets: false,
            proxy_mode: false,
            filter_open: None,
            filter_readlink: None,
            filter_stat: None,
            filter_ioctl: None,
            pinned_mappings: Vec::new(),
        }
    }
}

impl FileDescriptors {
    /// The first virtual file descriptor handed out for files.
    pub const FILE_D_BASE: i32 = 0x1000;
    /// Files the guest may have open at once (C++ has no limit). Guests
    /// leaking descriptors would otherwise exhaust the host's.
    pub const MAX_FILES: usize = 1024;

    /// Register an open file and return its virtual file descriptor, or
    /// `None` when the table is full.
    pub fn assign_file(&mut self, file: OpenFile) -> Option<i32> {
        if self.files.len() >= Self::MAX_FILES {
            return None;
        }
        // Skip numbers still in use after the counter wrapped.
        loop {
            let vfd = self.file_counter;
            self.file_counter = match self.file_counter.checked_add(1) {
                Some(n) => n,
                None => Self::FILE_D_BASE,
            };
            if let std::collections::btree_map::Entry::Vacant(e) = self.files.entry(vfd) {
                e.insert(file);
                return Some(vfd);
            }
        }
    }

    /// What `vfd` refers to: an open file, or one of the standard streams
    /// (C++ `translate`, which only lets 0, 1 and 2 through to the host).
    pub fn translate(&mut self, vfd: i32) -> Option<Target<'_>> {
        if let Some(f) = self.files.get_mut(&vfd) {
            return Some(Target::File(f));
        }
        (0..=2).contains(&vfd).then_some(Target::Stdio(vfd))
    }

    /// An open file (C++ `get`, which doesn't let the standard streams through).
    pub fn get(&mut self, vfd: i32) -> Option<&mut OpenFile> {
        self.files.get_mut(&vfd)
    }

    /// Close a virtual file descriptor (C++ `erase`). Returns false if it
    /// wasn't open.
    pub fn erase(&mut self, vfd: i32) -> bool {
        self.files.remove(&vfd).is_some()
    }

    /// Number of open files.
    pub fn len(&self) -> usize {
        self.files.len()
    }
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Writing to files is only allowed in proxy mode (C++ `permit_write`;
    /// sockets aren't ported yet).
    pub const fn permit_write(&self) -> bool {
        self.proxy_mode
    }
}

/// A host file handle for standard stream `n`, for `fstat`.
pub(crate) fn stdio_file(n: i32) -> io::Result<File> {
    use std::os::fd::AsFd;
    let owned = match n {
        0 => io::stdin().as_fd().try_clone_to_owned()?,
        1 => io::stdout().as_fd().try_clone_to_owned()?,
        _ => io::stderr().as_fd().try_clone_to_owned()?,
    };
    Ok(File::from(owned))
}

/// Size of the RISC-V `struct stat` (C++ `riscv_stat`).
pub(crate) const RISCV_STAT_SIZE: usize = 128;

/// Host metadata as the RISC-V `struct stat` (C++ `copy_stat_buffer`).
pub(crate) fn riscv_stat(md: &std::fs::Metadata) -> [u8; RISCV_STAT_SIZE] {
    use std::os::unix::fs::MetadataExt;
    let mut b = [0u8; RISCV_STAT_SIZE];
    let mut put = |ofs: usize, bytes: &[u8]| {
        if let Some(dst) = b.get_mut(ofs..ofs + bytes.len()) {
            dst.copy_from_slice(bytes);
        }
    };
    put(0, &md.dev().to_le_bytes());
    put(8, &md.ino().to_le_bytes());
    put(16, &md.mode().to_le_bytes());
    put(20, &(md.nlink() as u32).to_le_bytes());
    put(24, &md.uid().to_le_bytes());
    put(28, &md.gid().to_le_bytes());
    put(32, &md.rdev().to_le_bytes());
    put(48, &(md.size() as i64).to_le_bytes());
    put(56, &(md.blksize() as i32).to_le_bytes());
    put(64, &(md.blocks() as i64).to_le_bytes());
    put(72, &md.atime().to_le_bytes());
    put(80, &(md.atime_nsec() as u64).to_le_bytes());
    put(88, &md.mtime().to_le_bytes());
    put(96, &(md.mtime_nsec() as u64).to_le_bytes());
    put(104, &md.ctime().to_le_bytes());
    put(112, &(md.ctime_nsec() as u64).to_le_bytes());
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_streams_translate() {
        let mut fds = FileDescriptors::default();
        assert!(matches!(fds.translate(1), Some(Target::Stdio(1))));
        assert!(fds.translate(3).is_none());
        assert!(fds.translate(-100).is_none());
        assert!(fds.get(0).is_none());
        assert!(!fds.erase(0x1000));
    }

    #[test]
    fn files_get_virtual_numbers() {
        let mut fds = FileDescriptors::default();
        let file = File::open("/dev/null").unwrap();
        let vfd = fds
            .assign_file(OpenFile {
                file,
                path: "/dev/null".into(),
                access: 0,
            })
            .unwrap();
        assert_eq!(vfd, FileDescriptors::FILE_D_BASE);
        assert!(matches!(fds.translate(vfd), Some(Target::File(_))));
        assert!(fds.erase(vfd));
        assert!(fds.translate(vfd).is_none());
    }
}
