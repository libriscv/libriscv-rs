//! Port of `linux/syscalls_poll.cpp` and `linux/syscalls_select.cpp`.
//!
//! C++ forwards `ppoll` to the host with the translated descriptors. The
//! guest's descriptors are standard streams and regular files here, which
//! never block, so readiness is answered without asking the host:
//! descriptors that aren't open get `POLLNVAL`, the others are ready for
//! what was asked (stdin only for input, stdout and stderr only for
//! output). Nothing ever waits, so the timeout is not used.

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::syscalls::errno::{EINVAL, neg};
use crate::syscalls::fds::Target;

const POLLIN: i16 = 0x1;
const POLLOUT: i16 = 0x4;
const POLLNVAL: i16 = 0x20;
const MAX_POLL_FDS: u64 = 128;

/// ppoll(fds, nfds, timeout, sigmask)
pub(super) fn syscall_ppoll(m: &mut Machine) -> Result<()> {
    let g_fds = m.sysarg(0);
    let nfds = u64::from(m.sysarg(1) as u32);
    let g_ts = m.sysarg(2);
    // C++ always reads the timeout, and faults on a NULL one (an infinite
    // wait). That is a C++ bug; here NULL is accepted.
    if g_ts != 0 {
        let mut ts = [0u8; 16];
        m.copy_from_guest(&mut ts, g_ts)?;
    }
    if nfds > MAX_POLL_FDS {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    let Some(fds) = m.fds.as_deref_mut() else {
        // No file descriptors: nothing can become ready
        m.set_result(0);
        return Ok(());
    };
    let mut raw = vec![0u8; nfds as usize * 8];
    m.memory.memcpy_out(&mut raw, g_fds)?;
    let mut ready = 0u64;
    for pfd in raw.as_chunks_mut::<8>().0 {
        let [f0, f1, f2, f3, e0, e1, _, _] = *pfd;
        let fd = i32::from_le_bytes([f0, f1, f2, f3]);
        let events = i16::from_le_bytes([e0, e1]);
        let possible = match fds.translate(fd) {
            None => POLLNVAL,
            Some(Target::Stdio(0)) => POLLIN,
            Some(Target::Stdio(_)) => POLLOUT,
            Some(Target::File(_)) => POLLIN | POLLOUT,
        };
        let revents = if possible == POLLNVAL {
            POLLNVAL
        } else {
            possible & events
        };
        let [r0, r1] = revents.to_le_bytes();
        *pfd = [f0, f1, f2, f3, e0, e1, r0, r1];
        if revents != 0 {
            ready += 1;
        }
    }
    if ready > 0 {
        m.copy_to_guest(g_fds, &raw)?;
    }
    m.set_result(ready);
    Ok(())
}

/// pselect6: without file descriptors nothing can become ready (a
/// timeout); otherwise not implemented, as in C++.
pub(super) fn syscall_pselect(m: &mut Machine) -> Result<()> {
    if m.fds.is_none() {
        m.set_result(0);
        return Ok(());
    }
    Err(MachineError::new(
        ExceptionKind::SystemCallFailed,
        "pselect() not implemented",
        0,
    ))
}
