//! Port of `guest/guest_cpp_string.hpp`: libstdc++'s `std::string`.

use super::{
    GuestInit, GuestPtr, GuestType, MAX_BYTES, container_error, guest_alloc, guest_free,
    pass_by_value,
};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::vmcall::{ArgRegs, VmCallArg};

/// libstdc++'s `std::string` (C++ `GuestStdString`): `_M_p`, the size,
/// and a 16-byte union of the inline characters and the capacity. 32
/// bytes, aligned like a guest word.
///
/// The pointer decides where the characters are, not the size: libstdc++
/// points it at the inline buffer for a short string and at a heap block
/// otherwise, and a string that was allocated big and then shrunk keeps
/// its heap block. A null pointer is a host-side string with no guest
/// address yet, the only case where the characters are read from `data`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GuestStdString {
    /// `_M_p`
    pub ptr: u64,
    pub size: u64,
    /// The inline characters, or (first 8 bytes) the capacity.
    pub data: [u8; 16],
}

impl GuestPod for GuestStdString {
    const SIZE: usize = 32;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.ptr.to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.extend_from_slice(&self.data);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self {
            ptr: u64::read_guest(bytes),
            size: u64::read_guest(bytes.get(8..).unwrap_or_default()),
            data: <[u8; 16]>::read_guest(bytes.get(16..).unwrap_or_default()),
        }
    }
}

impl GuestStdString {
    /// The longest string kept inline.
    pub const SSO: u64 = 15;
    /// `offsetof(GuestStdString, data)`
    pub const DATA_OFFSET: u64 = 16;

    /// A host-side string with no guest address (C++
    /// `GuestStdString(machine, str)`): a short one is inline with a null
    /// pointer, a long one is copied to the native heap.
    pub fn new(m: &mut Machine, s: &str) -> Result<Self> {
        Self::from_bytes(m, 0, s.as_bytes())
    }

    /// A string that will live at the guest address `self_addr` (0: none
    /// yet), holding `bytes` (C++ `GuestStdString(machine, self, str)`).
    pub fn from_bytes(m: &mut Machine, self_addr: u64, bytes: &[u8]) -> Result<Self> {
        let mut s = Self::default();
        s.set_string(m, self_addr, bytes)?;
        Ok(s)
    }

    pub const fn empty(&self) -> bool {
        self.size == 0
    }

    /// The capacity, valid for a heap-allocated string.
    pub fn capacity(&self) -> u64 {
        u64::read_guest(&self.data)
    }

    /// The characters, wherever they live (C++ `to_view`). Fails, before
    /// touching guest memory, on a size above `max_len`.
    pub fn to_view<'a>(&'a self, m: &'a Machine, max_len: usize) -> Result<&'a [u8]> {
        if self.size > max_len as u64 {
            return Err(container_error("Guest std::string too large (size > 16MB)"));
        }
        if self.size == 0 {
            return Ok(&[]);
        }
        if self.ptr == 0 {
            // A size above SSO with a null pointer can't be inline.
            return self
                .data
                .get(..self.size as usize)
                .ok_or(container_error("Guest std::string: invalid inline size"));
        }
        m.memory.memview(self.ptr, self.size)
    }

    /// The characters as a host string (C++ `to_string`). Invalid UTF-8 is
    /// replaced, as by `Memory::memstring`; use [`to_view`](Self::to_view)
    /// for the bytes.
    pub fn to_string(&self, m: &Machine) -> Result<String> {
        Ok(String::from_utf8_lossy(self.to_view(m, MAX_BYTES)?).into_owned())
    }

    /// Replace the characters (C++ `set_string`), freeing a heap block the
    /// string owned. `self_addr` is where this string lives in guest memory
    /// (0: nowhere yet).
    pub fn set_string(&mut self, m: &mut Machine, self_addr: u64, bytes: &[u8]) -> Result<()> {
        self.free(m)?;
        let len = bytes.len() as u64;
        if len <= Self::SSO {
            // Leave ptr null when there is no guest address yet, instead
            // of pointing it into the guest's null page, so that to_view()
            // can tell an unplaced host-side string apart. relocate() fills
            // it in once there is an address.
            self.ptr = if self_addr != 0 {
                self_addr.wrapping_add(Self::DATA_OFFSET)
            } else {
                0
            };
            self.size = len;
            self.data = [0; 16];
            if let Some(d) = self.data.get_mut(..bytes.len()) {
                d.copy_from_slice(bytes);
            }
        } else {
            let ptr = guest_alloc(m, bytes.len() + 1)?;
            m.memory.memcpy(ptr, bytes)?;
            m.memory.write_u8(ptr.wrapping_add(len), 0)?;
            self.ptr = ptr;
            self.size = len;
            self.data = [0; 16];
            self.data[..8].copy_from_slice(&len.to_le_bytes());
        }
        Ok(())
    }

    /// The string now lives at `self_addr` (C++ `move`). Only for strings
    /// built by `set_string`, which never over-allocates: a guest string
    /// shrunk below SSO is still on the heap, and re-pointing it at the
    /// inline buffer would lose it.
    pub const fn relocate(&mut self, self_addr: u64) {
        if self.size <= Self::SSO {
            self.ptr = self_addr.wrapping_add(Self::DATA_OFFSET);
        }
    }

    /// Release the heap block (C++ `free`). Same caveat as
    /// [`relocate`](Self::relocate): a guest string shrunk below SSO owns a
    /// heap block this doesn't release.
    pub fn free(&mut self, m: &mut Machine) -> Result<()> {
        if self.size > Self::SSO {
            guest_free(m, self.ptr)?;
        }
        self.ptr = 0;
        self.size = 0;
        Ok(())
    }
}

impl GuestType for GuestStdString {
    const SELF_REFERENCING: bool = true;
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        self.free(m)
    }
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<Self>::new(addr);
        let mut s = ptr.get(m)?;
        s.free(m)?;
        ptr.set(m, &s)
    }
    fn relocate_at(m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<Self>::new(addr);
        let mut s = ptr.get(m)?;
        s.relocate(addr);
        ptr.set(m, &s)
    }
}

impl GuestInit<GuestStdString> for [u8] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let s = GuestStdString::from_bytes(m, addr, self)?;
        GuestPtr::new(addr).set(m, &s)
    }
}
impl GuestInit<GuestStdString> for str {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestStdString>::init_at(self.as_bytes(), m, addr)
    }
}
impl GuestInit<GuestStdString> for &str {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestStdString>::init_at(self.as_bytes(), m, addr)
    }
}
impl GuestInit<GuestStdString> for String {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestStdString>::init_at(self.as_bytes(), m, addr)
    }
}

impl GuestPtr<GuestStdString> {
    /// The characters as a host string.
    pub fn to_string(&self, m: &Machine) -> Result<String> {
        self.get(m)?.to_string(m)
    }
    /// The characters.
    pub fn to_bytes(&self, m: &Machine) -> Result<Vec<u8>> {
        Ok(self.get(m)?.to_view(m, super::MAX_BYTES)?.to_vec())
    }
    /// Replace the characters in place (C++ `set_string(machine, self,
    /// ...)`, and `ScopedArenaObject`'s string assignment).
    pub fn set_string(&self, m: &mut Machine, bytes: &[u8]) -> Result<()> {
        let mut s = self.get(m)?;
        s.set_string(m, self.address(), bytes)?;
        self.set(m, &s)
    }
}

/// By value: pushed onto the stack (C++ `setup_call`).
impl VmCallArg for GuestStdString {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        pass_by_value(self, m, regs)
    }
}
