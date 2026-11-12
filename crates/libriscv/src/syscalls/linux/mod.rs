//! Port of `linux/system_calls.cpp`: the Linux system call set for
//! regular (glibc, musl, Rust, Go, ...) RV64 executables.
//!
//! Guest input is untrusted. Everything here either completes, returns a
//! negative errno to the guest, or fails with a `MachineError` exactly
//! where C++ throws (mostly guest memory faults). Nothing touches the host
//! beyond what the file descriptor table permits, and host access goes
//! through `std` only. Deliberate differences from C++, all of them in
//! places where C++ hands the guest raw host access:
//!
//! * `fcntl` and `ioctl` are emulated instead of forwarded to the host
//!   with guest-chosen arguments: `ioctl` is denied (C++ denies all but
//!   terminal requests in proxy mode), `fcntl` answers the queries libc
//!   makes.
//! * `openat` refuses write access (`O_WRONLY`, `O_RDWR`, `O_CREAT`,
//!   `O_TRUNC`, `O_APPEND`) outside proxy mode, where C++ would create or
//!   truncate the host file and only refuse the writes afterwards.
//! * Paths are looked up on the host only through the filters: `fstatat`
//!   with a path needs `filter_stat` (or `filter_open`), `faccessat`
//!   needs `filter_open`, `readlinkat` needs `filter_readlink` (or proxy
//!   mode). C++ goes to the host when no filter is installed.
//! * `dup`/`dup3` return virtual descriptors (C++ returns host numbers),
//!   `pipe2` and `getdents64` are not supported (`ENOSYS`), and reading
//!   from fds 1 and 2 is `EBADF`. `getcwd` checks the buffer size.
//! * `getrusage` is always the deterministic, sandboxed answer.
//! * `ppoll` answers from the descriptor table instead of polling the host
//!   (see `poll.rs`).
//! * epoll, eventfd and sockets are Phase 5; until then they return
//!   `ENOSYS`.

mod mman;
mod poll;

use crate::memory::PAGE_SIZE;
use std::io::{IoSlice, IoSliceMut, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::SYSCALL_EBREAK_NR;
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::BRK_MAX;
use crate::syscalls::errno::{
    EACCES, EBADF, EINVAL, ENOENT, ENOMEM, ENOSYS, EPERM, ERANGE, io_error, neg,
};
use crate::syscalls::fds::{FileDescriptors, OpenFile, Target, riscv_stat, stdio_file};
use crate::threads::MAIN_THREAD_TID;

/// C++ `ANTI_FINGERPRINTING_MASK_MICROS`.
const MASK_MICROS: i64 = !0x3;
/// C++ `ANTI_FINGERPRINTING_MASK_NANOS`.
const MASK_NANOS: i64 = !0x3FFF;
/// The longest a guest can make the host sleep at once (C++ `NANOSLEEP_MAX_SECONDS`).
const NANOSLEEP_MAX_SECONDS: i64 = 1;
/// Maximum length of a single read from stdin.
const STDIN_READ_MAX: u64 = 16 << 20;

fn stub_zero(m: &mut Machine) -> Result<()> {
    m.set_result(0);
    Ok(())
}

fn stub_nosys(m: &mut Machine) -> Result<()> {
    m.set_result(neg(ENOSYS));
    Ok(())
}

fn syscall_exit(m: &mut Machine) -> Result<()> {
    m.stop();
    Ok(())
}

/// True if the guest gets exact host time (proxy mode).
fn proxy_mode(m: &Machine) -> bool {
    m.fds.as_ref().is_some_and(|f| f.proxy_mode)
}

/// A guest path argument. Paths must be UTF-8 (`None` otherwise).
fn guest_path(m: &Machine, addr: u64) -> Result<Option<String>> {
    let bytes = m
        .memory
        .memstring_bytes(addr, crate::memory::MEMSTRING_MAX)?;
    Ok(String::from_utf8(bytes).ok())
}

// --- Files and standard streams -------------------------------------------

fn syscall_read(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let address = m.sysarg(1);
    let len = m.sysarg(2);
    if vfd == 0 {
        // Standard input only through the stdin callback
        if len > STDIN_READ_MAX {
            m.set_result(neg(ENOMEM));
            return Ok(());
        }
        let mut buffer = vec![0u8; len as usize];
        let result = m.stdin_read(&mut buffer);
        if result > 0 {
            let n = (result as u64).min(len) as usize;
            m.copy_to_guest(address, buffer.get(..n).unwrap_or_default())?;
        }
        m.set_result(result as u64);
        return Ok(());
    }
    let Some(fds) = m.fds.as_deref_mut() else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let Some(Target::File(f)) = fds.translate(vfd) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let mut bufs: Vec<IoSliceMut<'_>> = m
        .memory
        .gather_writable_buffers(address, len, 512)?
        .into_iter()
        .map(IoSliceMut::new)
        .collect();
    let res = f.file.read_vectored(&mut bufs);
    m.set_result(match res {
        Ok(n) => n as u64,
        Err(e) => io_error(&e),
    });
    Ok(())
}

fn syscall_pread64(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let address = m.sysarg(1);
    let len = m.sysarg(2);
    let offset = m.sysarg(3);
    let Some(fds) = m.fds.as_deref_mut() else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let Some(Target::File(f)) = fds.translate(vfd) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    if offset > i64::MAX as u64 {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    // preadv into the gathered buffers.
    let mut total = 0u64;
    let mut result = None;
    for buf in m.memory.gather_writable_buffers(address, len, 512)? {
        match f.file.read_at(buf, offset.wrapping_add(total)) {
            Ok(n) => {
                total += n as u64;
                if n < buf.len() {
                    break;
                }
            }
            Err(e) => {
                if total == 0 {
                    result = Some(io_error(&e));
                }
                break;
            }
        }
    }
    m.set_result(result.unwrap_or(total));
    Ok(())
}

fn syscall_write(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let address = m.sysarg(1);
    let len = m.sysarg(2);
    // Zero-copy, in at most 64 buffers.
    if vfd == 1 || vfd == 2 {
        let machine: &Machine = m;
        for data in machine.memory.gather_buffers(address, len, 64)? {
            machine.print(data);
        }
        m.set_result(len);
        return Ok(());
    }
    let Some(fds) = m.fds.as_deref_mut().filter(|f| f.permit_write()) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let Some(Target::File(f)) = fds.translate(vfd) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let bufs: Vec<IoSlice<'_>> = m
        .memory
        .gather_buffers(address, len, 64)?
        .into_iter()
        .map(IoSlice::new)
        .collect();
    let result = match f.file.write_vectored(&bufs) {
        Ok(n) => n as u64,
        Err(e) => io_error(&e),
    };
    m.set_result(result);
    Ok(())
}

/// The guest `iovec` array at `addr`: `(base, len)` pairs.
fn read_iovecs(m: &Machine, addr: u64, count: usize) -> Result<Vec<(u64, u64)>> {
    let mut raw = vec![0u8; count * 16];
    m.copy_from_guest(&mut raw, addr)?;
    Ok(raw
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            let (a, b) = c.split_at(8);
            let word = |s: &[u8]| s.try_into().map_or(0, u64::from_le_bytes);
            (word(a), word(b))
        })
        .collect())
}

fn syscall_readv(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let iov_g = m.sysarg(1);
    let count = m.sysarg(2) as i32;
    if !(1..=128).contains(&count) {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    if vfd == 0 {
        // Standard input, through the stdin callback like read()
        let vecs = read_iovecs(m, iov_g, count as usize)?;
        let mut total = 0u64;
        for (base, len) in vecs {
            if len > STDIN_READ_MAX {
                m.set_result(neg(ENOMEM));
                return Ok(());
            }
            let mut buffer = vec![0u8; len as usize];
            let r = m.stdin_read(&mut buffer);
            if r <= 0 {
                break;
            }
            let n = (r as u64).min(len);
            m.copy_to_guest(base, buffer.get(..n as usize).unwrap_or_default())?;
            total += n;
            if n < len {
                break;
            }
        }
        m.set_result(total);
        return Ok(());
    }
    let is_file = m
        .fds
        .as_deref_mut()
        .is_some_and(|f| matches!(f.translate(vfd), Some(Target::File(_))));
    if !is_file {
        m.set_result(neg(EBADF));
        return Ok(());
    }
    let vecs = read_iovecs(m, iov_g, count as usize)?;
    let mut total = 0u64;
    for (base, len) in vecs {
        let Some(Target::File(f)) = m.fds.as_deref_mut().and_then(|f| f.translate(vfd)) else {
            break;
        };
        let mut bufs: Vec<IoSliceMut<'_>> = m
            .memory
            .gather_writable_buffers(base, len, 64)?
            .into_iter()
            .map(IoSliceMut::new)
            .collect();
        match f.file.read_vectored(&mut bufs) {
            Ok(n) => {
                total += n as u64;
                if (n as u64) < len {
                    break;
                }
            }
            Err(e) => {
                if total == 0 {
                    m.set_result(io_error(&e));
                    return Ok(());
                }
                break;
            }
        }
    }
    m.set_result(total);
    Ok(())
}

fn syscall_writev(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let iov_g = m.sysarg(1);
    let count = m.sysarg(2) as i32;
    if !(0..=256).contains(&count) {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    let stdio = vfd == 1 || vfd == 2;
    if !stdio {
        let is_file = m
            .fds
            .as_deref_mut()
            .filter(|f| f.permit_write())
            .is_some_and(|f| matches!(f.translate(vfd), Some(Target::File(_))));
        if !is_file {
            m.set_result(neg(EBADF));
            return Ok(());
        }
    }
    let vecs = read_iovecs(m, iov_g, count as usize)?;
    let mut total = 0u64;
    for (base, len) in vecs {
        if stdio {
            let machine: &Machine = m;
            for data in machine.memory.gather_buffers(base, len, 256)? {
                machine.print(data);
            }
            total += len;
            continue;
        }
        let Some(Target::File(f)) = m.fds.as_deref_mut().and_then(|f| f.translate(vfd)) else {
            break;
        };
        let bufs: Vec<IoSlice<'_>> = m
            .memory
            .gather_buffers(base, len, 256)?
            .into_iter()
            .map(IoSlice::new)
            .collect();
        match f.file.write_vectored(&bufs) {
            Ok(n) => {
                total += n as u64;
                if (n as u64) < len {
                    break;
                }
            }
            Err(e) => {
                if total == 0 {
                    m.set_result(io_error(&e));
                    return Ok(());
                }
                break;
            }
        }
    }
    m.set_result(total);
    Ok(())
}

fn syscall_lseek(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let offset = m.sysarg(1) as i64;
    let whence = m.sysarg(2) as i32;
    let Some(f) = m.fds.as_deref_mut().and_then(|f| f.get(vfd)) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let pos = match whence {
        0 if offset >= 0 => SeekFrom::Start(offset as u64),
        1 => SeekFrom::Current(offset),
        2 => SeekFrom::End(offset),
        _ => {
            m.set_result(neg(EINVAL));
            return Ok(());
        }
    };
    let res = f.file.seek(pos);
    m.set_result(match res {
        Ok(p) => p,
        Err(e) => io_error(&e),
    });
    Ok(())
}

/// The access mode and creation flags of `openat` (asm-generic values).
mod oflags {
    pub(super) const O_ACCMODE: i32 = 3;
    pub(super) const O_WRONLY: i32 = 1;
    pub(super) const O_RDWR: i32 = 2;
    pub(super) const O_CREAT: i32 = 0o100;
    pub(super) const O_EXCL: i32 = 0o200;
    pub(super) const O_TRUNC: i32 = 0o1000;
    pub(super) const O_APPEND: i32 = 0o2000;
}

/// Resolve a guest path against a directory descriptor: absolute paths
/// stand alone, relative ones need a directory opened by the guest
/// (`AT_FDCWD` is not a host directory, as in C++).
fn resolve_at(fds: &mut FileDescriptors, dir_fd: i32, path: &str) -> Option<String> {
    if path.starts_with('/') {
        return Some(path.to_owned());
    }
    match fds.get(dir_fd) {
        Some(dir) => Some(format!("{}/{path}", dir.path.trim_end_matches('/'))),
        None => None,
    }
}

fn syscall_openat(m: &mut Machine) -> Result<()> {
    use oflags::*;
    let dir_fd = m.sysarg(0) as i32;
    let g_path = m.sysarg(1);
    let flags = m.sysarg(2) as i32;
    let mode = m.sysarg(3) as u32;
    let path = guest_path(m, g_path)?;
    let Some(fds) = m.fds.as_deref_mut().filter(|f| f.permit_filesystem) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let Some(mut path) = path else {
        m.set_result(neg(ENOENT));
        return Ok(());
    };
    if let Some(filter) = fds.filter_open.as_mut()
        && !filter(&mut path)
    {
        m.set_result(neg(EPERM));
        return Ok(());
    }
    let wants_write = flags & O_ACCMODE != 0 || flags & (O_CREAT | O_TRUNC | O_APPEND) != 0;
    if wants_write && !fds.proxy_mode {
        m.set_result(neg(EACCES));
        return Ok(());
    }
    let Some(full) = resolve_at(fds, dir_fd, &path) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let access = flags & O_ACCMODE;
    let mut opts = std::fs::OpenOptions::new();
    opts.read(access != O_WRONLY)
        .write(access == O_WRONLY || access == O_RDWR)
        .append(flags & O_APPEND != 0)
        .truncate(flags & O_TRUNC != 0)
        .mode(mode & 0o7777);
    if flags & O_CREAT != 0 {
        if flags & O_EXCL != 0 {
            opts.create_new(true);
        } else {
            opts.create(true);
        }
    }
    let result = match opts.open(&full) {
        Ok(file) => {
            let entry = OpenFile {
                file,
                path: full,
                access,
            };
            fds.assign_file(entry)
                .map_or(neg(crate::syscalls::errno::EMFILE), |vfd| vfd as u64)
        }
        Err(e) => io_error(&e),
    };
    m.set_result(result);
    Ok(())
}

fn syscall_close(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let result = if (0..=2).contains(&vfd) {
        0
    } else {
        match m.fds.as_deref_mut().map(|fds| fds.erase(vfd)) {
            Some(true) => 0,
            _ => neg(EBADF),
        }
    };
    m.set_result(result);
    Ok(())
}

/// Duplicate an open file into a new virtual descriptor.
fn dup_file(fds: &mut FileDescriptors, vfd: i32) -> u64 {
    let Some(f) = fds.get(vfd) else {
        return neg(EBADF);
    };
    let entry = match f.file.try_clone() {
        Ok(file) => OpenFile {
            file,
            path: f.path.clone(),
            access: f.access,
        },
        Err(e) => return io_error(&e),
    };
    fds.assign_file(entry)
        .map_or(neg(crate::syscalls::errno::EMFILE), |v| v as u64)
}

fn syscall_dup(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let result = match m.fds.as_deref_mut() {
        Some(fds) => dup_file(fds, vfd),
        None => neg(EBADF),
    };
    m.set_result(result);
    Ok(())
}

fn syscall_dup3(m: &mut Machine) -> Result<()> {
    // Replacing a descriptor (or the standard streams) is not supported:
    // only dup3(old, new) where `new` would be a fresh descriptor anyway.
    let old = m.sysarg(0) as i32;
    let new = m.sysarg(1) as i32;
    let exists = m.fds.as_deref_mut().map(|fds| fds.get(old).is_some());
    let result = match exists {
        None | Some(false) => neg(EBADF),
        Some(true) if old == new => neg(EINVAL),
        Some(true) => neg(EPERM),
    };
    m.set_result(result);
    Ok(())
}

fn syscall_fcntl(m: &mut Machine) -> Result<()> {
    const F_DUPFD: i32 = 0;
    const F_GETFD: i32 = 1;
    const F_SETFD: i32 = 2;
    const F_GETFL: i32 = 3;
    const F_SETFL: i32 = 4;
    const F_DUPFD_CLOEXEC: i32 = 1030;
    const FD_CLOEXEC: u64 = 1;
    let vfd = m.sysarg(0) as i32;
    let cmd = m.sysarg(1) as i32;
    let Some(fds) = m.fds.as_deref_mut() else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let result = match fds.translate(vfd) {
        None => neg(EBADF),
        Some(target) => match cmd {
            F_DUPFD | F_DUPFD_CLOEXEC => dup_file(fds, vfd),
            // Guest files are always close-on-exec (there is no exec).
            F_GETFD => match target {
                Target::File(_) => FD_CLOEXEC,
                Target::Stdio(_) => 0,
            },
            F_SETFD | F_SETFL => 0,
            F_GETFL => match target {
                Target::File(f) => f.access as u64,
                Target::Stdio(0) => 0,
                Target::Stdio(_) => oflags::O_WRONLY as u64,
            },
            _ => neg(EINVAL),
        },
    };
    m.set_result(result);
    Ok(())
}

fn syscall_ioctl(m: &mut Machine) -> Result<()> {
    let req = m.sysarg(1);
    let Some(fds) = m.fds.as_deref_mut() else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    if let Some(filter) = fds.filter_ioctl.as_mut()
        && !filter(req)
    {
        m.set_result(neg(EPERM));
        return Ok(());
    }
    // No request reaches the host.
    m.set_result(neg(ENOSYS));
    Ok(())
}

fn syscall_readlinkat(m: &mut Machine) -> Result<()> {
    let dir_fd = m.sysarg(0) as i32;
    let g_path = m.sysarg(1);
    let g_buf = m.sysarg(2);
    let bufsize = m.sysarg(3);
    let path = guest_path(m, g_path)?;
    if bufsize > 1024 {
        m.set_result(neg(ENOMEM));
        return Ok(());
    }
    let Some(fds) = m.fds.as_deref_mut().filter(|f| f.permit_filesystem) else {
        m.set_result(neg(ENOSYS));
        return Ok(());
    };
    let Some(mut path) = path else {
        m.set_result(neg(ENOENT));
        return Ok(());
    };
    let answer: Vec<u8> = if let Some(filter) = fds.filter_readlink.as_mut() {
        if !filter(&mut path) {
            m.set_result(neg(EPERM));
            return Ok(());
        }
        // The filter always rewrites the answer
        path.into_bytes()
    } else if fds.proxy_mode {
        let Some(full) = resolve_at(fds, dir_fd, &path) else {
            m.set_result(neg(EBADF));
            return Ok(());
        };
        match std::fs::read_link(full) {
            Ok(target) => {
                use std::os::unix::ffi::OsStrExt;
                target.as_os_str().as_bytes().to_vec()
            }
            Err(e) => {
                m.set_result(io_error(&e));
                return Ok(());
            }
        }
    } else {
        m.set_result(neg(EPERM));
        return Ok(());
    };
    // Like the kernel, truncate to the buffer (C++ copies all of it).
    let n = answer.len().min(bufsize as usize);
    m.copy_to_guest(g_buf, answer.get(..n).unwrap_or_default())?;
    m.set_result(n as u64);
    Ok(())
}

fn syscall_getcwd(m: &mut Machine) -> Result<()> {
    let g_buf = m.sysarg(0);
    let size = m.sysarg(1);
    let Some(fds) = m.fds.as_deref() else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    if fds.cwd.is_empty() {
        m.set_result(u64::MAX);
        return Ok(());
    }
    let mut cwd = fds.cwd.clone().into_bytes();
    cwd.push(0);
    if (cwd.len() as u64) > size {
        m.set_result(neg(ERANGE));
        return Ok(());
    }
    m.copy_to_guest(g_buf, &cwd)?;
    m.set_result(cwd.len() as u64);
    Ok(())
}

/// Write host metadata as a RISC-V `struct stat`, or the error.
fn finish_stat(m: &mut Machine, g_buf: u64, md: std::io::Result<std::fs::Metadata>) -> Result<()> {
    match md {
        Ok(md) => {
            m.copy_to_guest(g_buf, &riscv_stat(&md))?;
            m.set_result(0);
        }
        Err(e) => m.set_result(io_error(&e)),
    }
    Ok(())
}

/// Metadata of an open descriptor (or a standard stream).
fn fd_metadata(fds: &mut FileDescriptors, vfd: i32) -> Option<std::io::Result<std::fs::Metadata>> {
    Some(match fds.translate(vfd)? {
        Target::File(f) => f.file.metadata(),
        Target::Stdio(n) => stdio_file(n).and_then(|f| f.metadata()),
    })
}

fn syscall_fstat(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let g_buf = m.sysarg(1);
    let Some(fds) = m.fds.as_deref_mut() else {
        m.set_result(neg(ENOSYS));
        return Ok(());
    };
    match fd_metadata(fds, vfd) {
        Some(md) => finish_stat(m, g_buf, md),
        None => {
            m.set_result(neg(EBADF));
            Ok(())
        }
    }
}

fn syscall_fstatat(m: &mut Machine) -> Result<()> {
    const AT_SYMLINK_NOFOLLOW: i32 = 0x100;
    let vfd = m.sysarg(0) as i32;
    let g_path = m.sysarg(1);
    let g_buf = m.sysarg(2);
    let flags = m.sysarg(3) as i32;
    let path = guest_path(m, g_path)?;
    let Some(fds) = m.fds.as_deref_mut().filter(|f| f.permit_filesystem) else {
        m.set_result(neg(ENOSYS));
        return Ok(());
    };
    let Some(path) = path else {
        m.set_result(neg(ENOENT));
        return Ok(());
    };
    if path.is_empty() {
        // fstat (AT_EMPTY_PATH)
        return match fd_metadata(fds, vfd) {
            Some(md) => finish_stat(m, g_buf, md),
            None => {
                m.set_result(neg(EBADF));
                Ok(())
            }
        };
    }
    let allowed = if let Some(filter) = fds.filter_stat.as_mut() {
        filter(&path)
    } else if let Some(filter) = fds.filter_open.as_mut() {
        let mut copy = path.clone();
        filter(&mut copy) && copy == path
    } else {
        fds.proxy_mode
    };
    if !allowed {
        m.set_result(neg(EPERM));
        return Ok(());
    }
    let Some(full) = resolve_at(fds, vfd, &path) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let md = if flags & AT_SYMLINK_NOFOLLOW != 0 {
        std::fs::symlink_metadata(full)
    } else {
        std::fs::metadata(full)
    };
    finish_stat(m, g_buf, md)
}

fn syscall_faccessat(m: &mut Machine) -> Result<()> {
    const R_OK: i32 = 4;
    const W_OK: i32 = 2;
    const X_OK: i32 = 1;
    let vfd = m.sysarg(0) as i32;
    let g_path = m.sysarg(1);
    let mode = m.sysarg(2) as i32;
    let path = guest_path(m, g_path)?;
    let Some(fds) = m.fds.as_deref_mut().filter(|f| f.permit_filesystem) else {
        m.set_result(neg(ENOSYS));
        return Ok(());
    };
    let Some(mut path) = path else {
        m.set_result(neg(ENOENT));
        return Ok(());
    };
    // Only paths the guest could open can be probed.
    let allowed = match fds.filter_open.as_mut() {
        Some(filter) => filter(&mut path),
        None => fds.proxy_mode,
    };
    if !allowed {
        m.set_result(neg(EACCES));
        return Ok(());
    }
    let Some(full) = resolve_at(fds, vfd, &path) else {
        m.set_result(neg(EBADF));
        return Ok(());
    };
    let result = match std::fs::metadata(&full) {
        Err(e) => io_error(&e),
        Ok(md) => {
            use std::os::unix::fs::PermissionsExt;
            let mut ok = true;
            if mode & R_OK != 0 {
                ok &= std::fs::File::open(&full).is_ok();
            }
            if mode & W_OK != 0 {
                ok &= fds.proxy_mode && std::fs::OpenOptions::new().write(true).open(&full).is_ok();
            }
            if mode & X_OK != 0 {
                ok &= md.permissions().mode() & 0o111 != 0;
            }
            if ok { 0 } else { neg(EACCES) }
        }
    };
    m.set_result(result);
    Ok(())
}

// --- Time -------------------------------------------------------------------

/// `(seconds, nanoseconds)` of clock `clkid`, or `None` for unknown
/// clocks. The monotonic clocks count from machine creation.
fn clock_time(m: &Machine, clkid: i32) -> Option<(i64, i64)> {
    let d = match clkid {
        // CLOCK_REALTIME, CLOCK_REALTIME_COARSE, CLOCK_TAI
        0 | 5 | 11 => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default(),
        // MONOTONIC, PROCESS_CPUTIME, THREAD_CPUTIME, MONOTONIC_RAW,
        // MONOTONIC_COARSE, BOOTTIME
        1 | 2 | 3 | 4 | 6 | 7 => m.epoch.elapsed(),
        _ => return None,
    };
    Some((
        i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        i64::from(d.subsec_nanos()),
    ))
}

fn timespec_bytes(sec: i64, nsec: i64) -> [u8; 16] {
    let mut b = [0u8; 16];
    let (s, n) = b.split_at_mut(8);
    s.copy_from_slice(&sec.to_le_bytes());
    n.copy_from_slice(&nsec.to_le_bytes());
    b
}

fn read_timespec(m: &Machine, addr: u64) -> Result<(i64, i64)> {
    let mut b = [0u8; 16];
    m.copy_from_guest(&mut b, addr)?;
    let (s, n) = b.split_at(8);
    let word = |s: &[u8]| s.try_into().map_or(0, i64::from_le_bytes);
    Ok((word(s), word(n)))
}

fn syscall_gettimeofday(m: &mut Machine) -> Result<()> {
    let buffer = m.sysarg(0);
    let (sec, nsec) = clock_time(m, 0).unwrap_or_default();
    let mut usec = nsec / 1000;
    if !proxy_mode(m) {
        usec &= MASK_MICROS;
    }
    m.copy_to_guest(buffer, &timespec_bytes(sec, usec))?;
    m.set_result(0);
    Ok(())
}

fn syscall_clock_gettime(m: &mut Machine) -> Result<()> {
    let clkid = m.sysarg(0) as i32;
    let buffer = m.sysarg(1);
    let Some((sec, mut nsec)) = clock_time(m, clkid) else {
        m.set_result(neg(EINVAL));
        return Ok(());
    };
    if !proxy_mode(m) {
        nsec &= MASK_NANOS;
    }
    m.copy_to_guest(buffer, &timespec_bytes(sec, nsec))?;
    m.set_result(0);
    Ok(())
}

/// Port of `nanosleep_clamp_and_penalize`: a guest must not block the host
/// for long, and sleeping doesn't advance the instruction counter. Clamp
/// the duration, and charge one instruction per microsecond requested.
/// Returns the clamped request.
fn clamp_and_penalize(m: &mut Machine, sec: i64, nsec: i64) -> (i64, i64) {
    let requested_ns = (sec as u64)
        .wrapping_mul(1_000_000_000)
        .wrapping_add(nsec as u64);
    m.penalize(requested_ns / 1000);
    let mut ts = (sec.min(NANOSLEEP_MAX_SECONDS), nsec);
    // Cooperative threads: sleeping blocks every guest thread. If another
    // can run, sleep only briefly and hand it the time.
    const SHARED_SLEEP_MAX_NS: i64 = 1_000_000;
    let others = m
        .threads()
        .is_some_and(|t| !t.suspended_threads().is_empty());
    if others && (ts.0 > 0 || ts.1 > SHARED_SLEEP_MAX_NS) {
        ts = (0, SHARED_SLEEP_MAX_NS);
    }
    ts
}

/// Sleep on the host like `nanosleep`: `EINVAL` for invalid nanoseconds.
fn host_sleep((sec, nsec): (i64, i64)) -> u64 {
    if !(0..1_000_000_000).contains(&nsec) {
        return neg(EINVAL);
    }
    std::thread::sleep(Duration::new(sec as u64, nsec as u32));
    0
}

/// Read and validate a sleep request, masking it outside proxy mode.
fn sleep_request(m: &Machine, addr: u64) -> Result<Option<(i64, i64)>> {
    let (sec, nsec) = read_timespec(m, addr)?;
    if sec < 0 || nsec < 0 {
        return Ok(None);
    }
    let nsec = if proxy_mode(m) {
        nsec
    } else {
        nsec & MASK_NANOS
    };
    Ok(Some((sec, nsec)))
}

fn syscall_nanosleep(m: &mut Machine) -> Result<()> {
    let g_req = m.sysarg(0);
    let g_rem = m.sysarg(1);
    let Some(req) = sleep_request(m, g_req)? else {
        m.set_result(neg(EINVAL));
        return Ok(());
    };
    let req = clamp_and_penalize(m, req.0, req.1);
    let mut rem = [0u8; 16];
    if g_rem != 0 {
        m.copy_from_guest(&mut rem, g_rem)?;
    }
    let result = host_sleep(req);
    if result == 0 {
        // C++ writes the clamped request back, and the remainder unchanged.
        m.copy_to_guest(g_req, &timespec_bytes(req.0, req.1))?;
        if g_rem != 0 {
            m.copy_to_guest(g_rem, &rem)?;
        }
    }
    // Hand the time over to the other guest threads.
    m.set_result(result);
    m.suspend_and_yield(result as i64);
    Ok(())
}

fn syscall_clock_nanosleep(m: &mut Machine) -> Result<()> {
    let g_req = m.sysarg(2);
    let g_rem = m.sysarg(3);
    let Some(req) = sleep_request(m, g_req)? else {
        m.set_result(neg(EINVAL));
        return Ok(());
    };
    let req = clamp_and_penalize(m, req.0, req.1);
    let result = host_sleep(req);
    if result == 0 && g_rem != 0 {
        m.copy_to_guest(g_rem, &timespec_bytes(0, 0))?;
    }
    m.set_result(result);
    m.suspend_and_yield(result as i64);
    Ok(())
}

// --- Process and system information -----------------------------------------

fn syscall_getrusage(m: &mut Machine) -> Result<()> {
    let who = m.sysarg(0) as i32;
    let g_usage = m.sysarg(1);
    // RUSAGE_SELF, RUSAGE_CHILDREN, RUSAGE_THREAD
    if !(-1..=1).contains(&who) {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    // Deterministic progress: the instruction counter.
    let micros = m.instruction_counter() / 1000;
    let mut usage = [0u8; 18 * 8];
    let mut put = |i: usize, v: u64| {
        if let Some(d) = usage.get_mut(i * 8..i * 8 + 8) {
            d.copy_from_slice(&v.to_le_bytes());
        }
    };
    put(0, micros / 1_000_000);
    put(1, micros % 1_000_000);
    // ru_maxrss: the pages in use, in KiB.
    put(
        4,
        (m.memory.pages_active() as u64).wrapping_mul(PAGE_SIZE) / 1024,
    );
    if g_usage != 0 {
        m.copy_to_guest(g_usage, &usage)?;
    }
    m.set_result(0);
    Ok(())
}

fn syscall_getpid(m: &mut Machine) -> Result<()> {
    m.set_result(MAIN_THREAD_TID as u64);
    Ok(())
}

fn syscall_getppid(m: &mut Machine) -> Result<()> {
    m.set_result(1);
    Ok(())
}

fn syscall_sched_getaffinity(m: &mut Machine) -> Result<()> {
    let cpusetsize = m.sysarg(1);
    let g_mask = m.sysarg(2);
    if cpusetsize < 8 || g_mask == 0 {
        m.set_result(neg(EINVAL));
    } else {
        m.copy_to_guest(g_mask, &1u64.to_le_bytes())?;
        m.set_result(8);
    }
    Ok(())
}

fn syscall_uname(m: &mut Machine) -> Result<()> {
    const UTSLEN: usize = 65;
    let buffer = m.sysarg(0);
    // The same answer as C++, so that guests behave identically.
    let fields = [
        "RISC-V C++ Emulator",
        "libriscv",
        "5.6.0",
        "",
        "rv64imafdc",
        "(none)",
    ];
    let mut uts = [0u8; UTSLEN * 6];
    for (i, field) in fields.iter().enumerate() {
        if let Some(dst) = uts.get_mut(i * UTSLEN..i * UTSLEN + field.len()) {
            dst.copy_from_slice(field.as_bytes());
        }
    }
    m.copy_to_guest(buffer, &uts)?;
    m.set_result(0);
    Ok(())
}

fn syscall_capget(m: &mut Machine) -> Result<()> {
    let header_ptr = m.sysarg(0);
    let data_ptr = m.sysarg(1);
    let mut header = [0u8; 8];
    m.copy_from_guest(&mut header, header_ptr)?;
    let version = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if version != 0x2008_0522 {
        m.set_result(neg(EINVAL));
    } else {
        // No capabilities
        m.copy_to_guest(data_ptr, &[0u8; 12])?;
        m.set_result(0);
    }
    Ok(())
}

fn syscall_brk(m: &mut Machine) -> Result<()> {
    let heap = m.memory.heap_address();
    let new_end = m.sysarg(0).min(heap.wrapping_add(BRK_MAX));
    // Static glibc puts the TLS block at the first break.
    if new_end >= heap {
        m.memory.set_brk_address(new_end);
    }
    m.set_result(m.memory.brk_address());
    Ok(())
}

/// Fill `buf` with random bytes from the host.
pub(crate) fn host_random(buf: &mut [u8]) -> bool {
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(buf))
        .is_ok()
}

fn syscall_getrandom(m: &mut Machine) -> Result<()> {
    let g_addr = m.sysarg(0);
    let g_len = m.sysarg(1);
    if g_len > 256 {
        // As C++: -1, not an errno.
        m.set_result(u64::MAX);
        return Ok(());
    }
    let mut buffer = vec![0u8; g_len as usize];
    if !host_random(&mut buffer) {
        m.set_result(u64::MAX);
        return Ok(());
    }
    if g_len > 0 {
        m.copy_to_guest(g_addr, &buffer)?;
        // getrandom() is a slow system call: 20k instructions per byte
        m.penalize(20_000 * g_len);
    }
    m.set_result(g_len);
    Ok(())
}

fn syscall_riscv_flush_icache(m: &mut Machine) -> Result<()> {
    let begin = m.sysarg(0);
    let end = m.sysarg(1);
    if begin < end {
        let penalty = m.memory.flush_execute_segments(begin, end);
        m.penalize(penalty);
    } else {
        m.memory.mark_execute_segments_stale();
    }
    m.set_result(0);
    Ok(())
}

// --- Signals ----------------------------------------------------------------

const LINUX_SA_ONSTACK: u64 = 0x0800_0000;

fn syscall_sigaltstack(m: &mut Machine) -> Result<()> {
    use crate::signals::SignalStack;
    let ss = m.sysarg(0);
    let old_ss = m.sysarg(1);
    let tid = m.gettid();
    let current = m.signals().per_thread(tid).stack;
    if old_ss != 0 {
        m.copy_to_guest(old_ss, &current.to_bytes())?;
    }
    if ss != 0 {
        let mut raw = [0u8; SignalStack::SIZE];
        m.copy_from_guest(&mut raw, ss)?;
        m.signals().per_thread(tid).stack = SignalStack::from_bytes(&raw);
    }
    m.set_result(0);
    Ok(())
}

fn syscall_sigaction(m: &mut Machine) -> Result<()> {
    let sig = m.sysarg(0) as i32;
    let action = m.sysarg(1);
    let old_action = m.sysarg(2);
    if sig == 0 {
        return Ok(());
    }
    let current = *m.sigaction(sig)?;
    // struct kernel_sigaction { handler; flags; mask; }
    if old_action != 0 {
        let mut sa = [0u8; 24];
        let flags = if current.altstack {
            LINUX_SA_ONSTACK
        } else {
            0
        };
        for (i, v) in [current.handler & !0xF, flags, u64::from(current.mask)]
            .into_iter()
            .enumerate()
        {
            if let Some(d) = sa.get_mut(i * 8..i * 8 + 8) {
                d.copy_from_slice(&v.to_le_bytes());
            }
        }
        m.copy_to_guest(old_action, &sa)?;
    }
    if action != 0 {
        let mut sa = [0u8; 24];
        m.copy_from_guest(&mut sa, action)?;
        let word = |i: usize| {
            sa.get(i * 8..i * 8 + 8)
                .and_then(|s| s.try_into().ok())
                .map_or(0, u64::from_le_bytes)
        };
        let act = m.sigaction(sig)?;
        act.handler = word(0);
        act.altstack = word(1) & LINUX_SA_ONSTACK != 0;
        act.mask = word(2) as u32;
    }
    m.set_result(0);
    Ok(())
}

fn syscall_rt_sigreturn(m: &mut Machine) -> Result<()> {
    m.leave_signal()
}

fn syscall_tkill(m: &mut Machine) -> Result<()> {
    let sig = m.sysarg(1) as i32;
    // A signal of zero, or without a handler, is ignored
    if sig == 0 || m.sigaction(sig)?.is_unset() {
        return Ok(());
    }
    m.enter_signal(sig)
}

fn syscall_prlimit64_stub(m: &mut Machine) -> Result<()> {
    const RLIMIT_STACK: i32 = 3;
    let resource = m.sysarg(1) as i32;
    let old_addr = m.sysarg(3);
    if old_addr != 0 {
        let v: u64 = if resource == RLIMIT_STACK {
            0x20_0000
        } else {
            0
        };
        let mut lim = [0u8; 16];
        let (cur, max) = lim.split_at_mut(8);
        cur.copy_from_slice(&v.to_le_bytes());
        max.copy_from_slice(&v.to_le_bytes());
        m.copy_to_guest(old_addr, &lim)?;
        m.set_result(0);
    } else {
        m.set_result(neg(EINVAL));
    }
    Ok(())
}

impl Machine {
    /// The file descriptor table (C++ `fds()`), present after
    /// `setup_linux_syscalls(true, ..)`.
    pub fn fds(&self) -> Option<&FileDescriptors> {
        self.fds.as_deref()
    }
    pub fn fds_mut(&mut self) -> Option<&mut FileDescriptors> {
        self.fds.as_deref_mut()
    }
    /// C++ `has_file_descriptors`.
    pub const fn has_file_descriptors(&self) -> bool {
        self.fds.is_some()
    }

    /// Install the newlib system call set (C++ `setup_newlib_syscalls`),
    /// with a file descriptor table if `filesystem` is set.
    pub fn setup_newlib_syscalls(&mut self, filesystem: bool) {
        self.install_syscall_handlers([
            (57, stub_zero as crate::SyscallHandler), // close
            (62, syscall_lseek),
            (63, syscall_read),
            (64, syscall_write),
            (80, stub_nosys), // fstat
            (93, syscall_exit),
            (169, syscall_gettimeofday),
            (214, syscall_brk),
            (403, syscall_clock_gettime),
        ]);
        if filesystem {
            self.fds = Some(Box::default());
        }
    }

    /// Install the Linux system call set (C++ `setup_linux_syscalls`).
    /// With `filesystem` or `sockets`, the machine gets a file descriptor
    /// table; the guest still can't open anything until
    /// `fds_mut().permit_filesystem` is set. Sockets are not ported yet.
    /// Combine with [`setup_linux`](Self::setup_linux) for the initial
    /// stack, and [`setup_posix_threads`](Self::setup_posix_threads) for
    /// threads.
    pub fn setup_linux_syscalls(&mut self, filesystem: bool, sockets: bool) {
        use crate::SyscallHandler as H;
        let handlers: &[(usize, H)] = &[
            (SYSCALL_EBREAK_NR, super::minimal::syscall_ebreak),
            (17, syscall_getcwd),
            // eventfd2, epoll_create, epoll_ctl, epoll_pwait: Phase 5
            (19, stub_nosys),
            (20, stub_nosys),
            (21, stub_nosys),
            (22, stub_nosys),
            (23, syscall_dup),
            (24, syscall_dup3),
            (25, syscall_fcntl),
            (29, syscall_ioctl),
            (48, syscall_faccessat),
            (56, syscall_openat),
            (57, syscall_close),
            (59, stub_nosys), // pipe2
            (61, stub_nosys), // getdents64
            (62, syscall_lseek),
            (63, syscall_read),
            (64, syscall_write),
            (65, syscall_readv),
            (66, syscall_writev),
            (67, syscall_pread64),
            (72, poll::syscall_pselect),
            (73, poll::syscall_ppoll),
            (78, syscall_readlinkat),
            (79, syscall_fstatat),
            (80, syscall_fstat),
            (90, syscall_capget),
            (93, syscall_exit),
            (94, syscall_exit), // exit_group
            (101, syscall_nanosleep),
            (113, syscall_clock_gettime),
            (403, syscall_clock_gettime),
            (114, stub_nosys), // clock_getres
            (115, syscall_clock_nanosleep),
            (123, syscall_sched_getaffinity),
            (130, syscall_tkill),
            (132, syscall_sigaltstack),
            (134, syscall_sigaction),
            (135, stub_zero), // rt_sigprocmask
            (139, syscall_rt_sigreturn),
            (160, syscall_uname),
            (167, stub_nosys), // prctl
            (169, syscall_gettimeofday),
            (165, syscall_getrusage),
            (172, syscall_getpid),
            (173, syscall_getppid),
            (174, stub_zero), // getuid
            (175, stub_zero), // geteuid
            (176, stub_zero), // getgid
            (177, stub_zero), // getegid
            (214, syscall_brk),
            (215, mman::syscall_munmap),
            (216, mman::syscall_mremap),
            (222, mman::syscall_mmap),
            (226, mman::syscall_mprotect),
            (227, stub_zero), // msync
            (233, mman::syscall_madvise),
            (258, stub_zero), // riscv_hwprobe
            (259, syscall_riscv_flush_icache),
            (278, syscall_getrandom),
            (291, stub_nosys), // statx
            (293, stub_nosys), // rseq
        ];
        self.install_syscall_handlers(handlers.iter().copied());
        // Without threads, stubs for what libc's thread setup needs
        if !self.has_threads() {
            self.install_syscall_handlers([
                (96, stub_zero as H), // set_tid_address
                (99, stub_zero),      // set_robust_list
                (261, syscall_prlimit64_stub),
            ]);
        }
        if filesystem || sockets {
            self.fds = Some(Box::default());
        }
    }
}
