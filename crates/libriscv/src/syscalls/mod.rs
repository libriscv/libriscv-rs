//! System call sets. Port of `posix/minimal.cpp`, `linux/system_calls.cpp`
//! and `posix/filedesc.hpp`.

pub(crate) mod errno;
pub(crate) mod fds;
mod linux;
mod minimal;

pub(crate) use linux::host_random;
