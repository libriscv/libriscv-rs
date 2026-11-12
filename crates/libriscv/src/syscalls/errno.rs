//! Linux errno values (asm-generic, as used by RISC-V) and helpers for
//! returning them from system calls.

pub(crate) const EPERM: i64 = 1;
pub(crate) const ENOENT: i64 = 2;
pub(crate) const EIO: i64 = 5;
pub(crate) const EBADF: i64 = 9;
pub(crate) const EAGAIN: i64 = 11;
pub(crate) const ENOMEM: i64 = 12;
pub(crate) const EACCES: i64 = 13;
pub(crate) const EEXIST: i64 = 17;
pub(crate) const EINVAL: i64 = 22;
pub(crate) const EMFILE: i64 = 24;
pub(crate) const ENOSPC: i64 = 28;
pub(crate) const ERANGE: i64 = 34;
pub(crate) const ENOSYS: i64 = 38;

/// A negative errno as a register value.
pub(crate) const fn neg(errno: i64) -> u64 {
    errno.wrapping_neg() as u64
}

/// The negative errno for a host I/O error (C++ `set_result_or_error`
/// returning `-errno`). The port only runs on Linux hosts, where host and
/// guest errno values agree.
pub(crate) fn io_error(e: &std::io::Error) -> u64 {
    neg(e.raw_os_error().map_or(EIO, i64::from))
}
