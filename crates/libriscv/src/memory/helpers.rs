//! Port of the host-side guest memory helpers in `memory_helpers_paging.hpp`
//! (`membuffer`, `memstring_view`, `strlen`, `memarray`, `try_memarray`,
//! `memspan`, `try_memmove`, both `memcmp`s and the machine-to-machine
//! `memcpy`) and of `util/buffer.hpp` (`riscv::Buffer`).
//!
//! C++ hands out raw pointers into guest memory (`T*`, `std::span<T>`,
//! `std::array<T, N>*`). Safe Rust can't reinterpret guest bytes as `T`, so
//! the typed views here ([`GuestSlice`], [`GuestSliceMut`]) wrap the
//! zero-copy byte view and decode/encode elements through [`GuestPod`] on
//! access. They still alias guest memory: writes through a
//! [`GuestSliceMut`] are visible to the guest, exactly like writes through
//! C++'s pointer.

use std::cmp::Ordering;
use std::marker::PhantomData;

use super::paging::{PAGE, PAGE_MASK};
use super::{Memory, PAGE_SIZE, RWREAD_BEGIN, protection_fault};
use crate::error::Result;

/// C++'s default `maxlen` for `memview`, `membuffer`, `memspan` and
/// `memarray`: 16 MiB.
pub const MEMVIEW_MAX: u64 = 16 << 20;

/// A value with a fixed guest memory layout (little-endian, C layout).
///
/// Implemented for the integer and floating-point types and arrays of
/// them. Implement it for `#[repr(C)]`-like structs by writing and reading
/// each field (and any padding) in order. Used to pass values to the guest
/// ([`VmCallArg`](crate::VmCallArg), [`Machine::stack_push_value`](crate::Machine::stack_push_value))
/// and to read them back ([`Memory::memarray`], [`Memory::memspan`],
/// [`ByPointer`](crate::ByPointer)).
pub trait GuestPod: Sized {
    /// `sizeof(T)` in the guest.
    const SIZE: usize;
    /// `alignof(T)` in the guest. The typed views fault on addresses that
    /// aren't a multiple of it, as C++ does.
    const ALIGN: usize;
    /// Append the value's guest representation (exactly [`SIZE`](Self::SIZE)
    /// bytes).
    fn write_guest(&self, out: &mut Vec<u8>);
    /// Decode a value from its guest representation. `bytes` holds at least
    /// [`SIZE`](Self::SIZE) bytes.
    fn read_guest(bytes: &[u8]) -> Self;
    /// Encode the value into `dst` (at least [`SIZE`](Self::SIZE) bytes).
    fn store_guest(&self, dst: &mut [u8]) {
        let mut out = Vec::with_capacity(Self::SIZE);
        self.write_guest(&mut out);
        let n = out.len().min(dst.len());
        if let (Some(d), Some(s)) = (dst.get_mut(..n), out.get(..n)) {
            d.copy_from_slice(s);
        }
    }
}

macro_rules! pod {
    ($($t:ty),*) => {$(
        impl GuestPod for $t {
            const SIZE: usize = size_of::<$t>();
            const ALIGN: usize = size_of::<$t>();
            fn write_guest(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn read_guest(bytes: &[u8]) -> Self {
                bytes
                    .first_chunk::<{ size_of::<$t>() }>()
                    .map_or(<$t>::default(), |b| <$t>::from_le_bytes(*b))
            }
            fn store_guest(&self, dst: &mut [u8]) {
                if let Some(d) = dst.first_chunk_mut::<{ size_of::<$t>() }>() {
                    *d = self.to_le_bytes();
                }
            }
        }
    )*};
}
pod!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl<T: GuestPod, const N: usize> GuestPod for [T; N] {
    const SIZE: usize = T::SIZE * N;
    const ALIGN: usize = T::ALIGN;
    fn write_guest(&self, out: &mut Vec<u8>) {
        for v in self {
            v.write_guest(out);
        }
    }
    fn read_guest(bytes: &[u8]) -> Self {
        std::array::from_fn(|i| {
            T::read_guest(bytes.get(i.saturating_mul(T::SIZE)..).unwrap_or_default())
        })
    }
}

/// Port of `riscv::Buffer`: guest memory as a list of zero-copy pieces,
/// which may or may not be one contiguous slice. Adjacent pieces of the
/// flat arena are always merged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Buffer<'a> {
    pieces: Vec<&'a [u8]>,
    len: usize,
}

impl<'a> Buffer<'a> {
    fn append(&mut self, piece: &'a [u8]) {
        self.len += piece.len();
        self.pieces.push(piece);
    }

    /// The whole buffer is one slice (C++ `is_sequential`).
    pub fn is_sequential(&self) -> bool {
        self.pieces.len() <= 1
    }
    /// The first piece: all of it when [`is_sequential`](Self::is_sequential)
    /// (C++ `strview`).
    pub fn strview(&self) -> &'a [u8] {
        self.pieces.first().copied().unwrap_or_default()
    }
    /// Total length in bytes (C++ `size`).
    pub const fn len(&self) -> usize {
        self.len
    }
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The pieces, in address order (C++ `foreach`).
    pub fn pieces(&self) -> &[&'a [u8]] {
        &self.pieces
    }
    /// Port of `Buffer::copy_to(char*, size_t)`: copy as many whole pieces
    /// as fit in `dst`, returning the bytes copied. Nothing is copied if
    /// the first piece doesn't fit.
    pub fn copy_to(&self, dst: &mut [u8]) -> usize {
        let mut len = 0;
        for piece in &self.pieces {
            let Some(d) = dst.get_mut(len..len + piece.len()) else {
                break;
            };
            d.copy_from_slice(piece);
            len += piece.len();
        }
        len
    }
    /// Append the whole buffer to `vec` (C++ `copy_to(std::vector&)`).
    pub fn copy_to_vec(&self, vec: &mut Vec<u8>) {
        vec.reserve(self.len);
        for piece in &self.pieces {
            vec.extend_from_slice(piece);
        }
    }
    /// The whole buffer as one vector.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut v = Vec::new();
        self.copy_to_vec(&mut v);
        v
    }
    /// The whole buffer as a (lossy) UTF-8 string (C++ `to_string`).
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.to_vec()).into_owned()
    }
}

/// A read-only typed view of guest memory (C++ `const T*` from `memarray`,
/// `std::span<const T>` from `memspan`). Elements are decoded on access.
#[derive(Debug, Clone, Copy)]
pub struct GuestSlice<'a, T> {
    bytes: &'a [u8],
    _t: PhantomData<fn() -> T>,
}

impl<'a, T: GuestPod> GuestSlice<'a, T> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            _t: PhantomData,
        }
    }
    /// Number of elements.
    pub const fn len(&self) -> usize {
        match self.bytes.len().checked_div(T::SIZE) {
            Some(n) => n,
            None => 0,
        }
    }
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Element `i`, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<T> {
        let start = i.checked_mul(T::SIZE)?;
        self.bytes
            .get(start..start.checked_add(T::SIZE)?)
            .map(T::read_guest)
    }
    /// The elements, in order.
    pub fn iter(&self) -> impl Iterator<Item = T> + 'a {
        let bytes = self.bytes;
        (0..self.len()).filter_map(move |i| Self::new(bytes).get(i))
    }
    /// Copy the elements out.
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().collect()
    }
    /// The underlying guest bytes.
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// A writable typed view of guest memory (C++ `T*` from `memarray`,
/// `std::span<T>` from `memspan`). Writes go straight to guest memory.
#[derive(Debug)]
pub struct GuestSliceMut<'a, T> {
    bytes: &'a mut [u8],
    _t: PhantomData<fn() -> T>,
}

impl<'a, T: GuestPod> GuestSliceMut<'a, T> {
    const fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes,
            _t: PhantomData,
        }
    }
    /// Number of elements.
    pub const fn len(&self) -> usize {
        match self.bytes.len().checked_div(T::SIZE) {
            Some(n) => n,
            None => 0,
        }
    }
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Element `i`, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<T> {
        GuestSlice::<T>::new(self.bytes).get(i)
    }
    /// Store `value` as element `i`. Returns false (and writes nothing) past
    /// the end.
    pub fn set(&mut self, i: usize, value: &T) -> bool {
        let Some(start) = i.checked_mul(T::SIZE) else {
            return false;
        };
        let Some(end) = start.checked_add(T::SIZE) else {
            return false;
        };
        match self.bytes.get_mut(start..end) {
            Some(d) => {
                value.store_guest(d);
                true
            }
            None => false,
        }
    }
    /// Store all of `values`, which must have [`len`](Self::len) elements.
    /// Returns false (and writes nothing) otherwise.
    pub fn copy_from_slice(&mut self, values: &[T]) -> bool {
        if values.len() != self.len() {
            return false;
        }
        for (i, v) in values.iter().enumerate() {
            self.set(i, v);
        }
        true
    }
    /// Copy the elements out.
    pub fn to_vec(&self) -> Vec<T> {
        GuestSlice::<T>::new(self.bytes).to_vec()
    }
    /// A read-only view of the same memory.
    pub fn as_slice(&self) -> GuestSlice<'_, T> {
        GuestSlice::new(self.bytes)
    }
    /// The underlying guest bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes
    }
    /// The underlying guest bytes, writable.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.bytes
    }
}

/// `count * T::SIZE`, faulting at `addr` when it would exceed `maxbytes`
/// (C++ `count > maxbytes / sizeof(T)`, so the multiplication can't
/// overflow).
fn array_bytes<T: GuestPod>(addr: u64, count: u64, maxbytes: u64) -> Result<u64> {
    let size = T::SIZE as u64;
    if maxbytes.checked_div(size).is_some_and(|max| count > max) {
        return Err(protection_fault(addr));
    }
    Ok(count.wrapping_mul(size))
}

/// `addr % alignof(T) != 0`.
fn misaligned<T: GuestPod>(addr: u64) -> bool {
    addr.checked_rem(T::ALIGN as u64).is_some_and(|r| r != 0)
}

impl Memory {
    /// The arena's bytes `[addr, addr + len)`, unchecked against the read
    /// and write windows (the callers check what C++ checks).
    fn arena_range(&self, addr: u64, len: u64) -> Option<&[u8]> {
        let a = usize::try_from(addr).ok()?;
        let n = usize::try_from(len).ok()?;
        self.arena.get(a..a.checked_add(n)?)
    }

    /// Port of `Memory::membuffer`: `[addr, addr + len)` as a [`Buffer`].
    /// A range inside the arena is one piece; otherwise it is gathered page
    /// by page, faulting at the first unreadable page. A `len` above
    /// `maxlen` (C++ default: [`MEMVIEW_MAX`]) is a protection fault.
    ///
    /// Like C++, the arena fast path only checks the arena's size, not its
    /// read window or page attributes.
    pub fn membuffer(&self, addr: u64, len: u64, maxlen: u64) -> Result<Buffer<'_>> {
        let mut result = Buffer::default();
        if len > maxlen {
            return Err(protection_fault(addr));
        }
        if len == 0 {
            return Ok(result);
        }
        let end = addr.wrapping_add(len);
        if end < self.arena_size
            && end > addr
            && let Some(piece) = self.arena_range(addr, len)
        {
            result.append(piece);
            return Ok(result);
        }
        for piece in self.gather_buffers(addr, len, usize::MAX)? {
            result.append(piece);
        }
        Ok(result)
    }

    /// Port of `Memory::strlen`: the length of the NUL-terminated string at
    /// `addr`, at most `maxlen` (C++ default [`MEMSTRING_MAX`](super::MEMSTRING_MAX)).
    /// Reads whole pages, with read permission checks.
    pub fn strlen(&self, addr: u64, maxlen: usize) -> Result<usize> {
        let mut addr = addr;
        let mut len = 0usize;
        loop {
            let offset = (addr & PAGE_MASK) as usize;
            let page = self.get_readable_pageno(addr / PAGE_SIZE)?;
            let chunk = page.get(offset..).unwrap_or_default();
            let max_bytes = PAGE - offset;
            let thislen = chunk.iter().position(|&b| b == 0).unwrap_or(max_bytes);
            len = len.saturating_add(thislen);
            if thislen != max_bytes {
                break;
            }
            addr = addr.wrapping_add(thislen as u64);
            if len >= maxlen {
                break;
            }
        }
        Ok(len.min(maxlen))
    }

    /// Port of `Memory::memstring_view`: a zero-copy view of the
    /// NUL-terminated string at `addr` (without the terminator), at most
    /// `maxlen` bytes (C++ default [`MEMSTRING_MAX`](super::MEMSTRING_MAX)).
    /// Only strings starting inside the flat arena can be viewed; anything
    /// else is a protection fault. Like C++, the arena's read window and
    /// page attributes aren't checked.
    pub fn memstring_view(&self, addr: u64, maxlen: usize) -> Result<&[u8]> {
        if addr < self.arena_size {
            let limit = (maxlen as u64).min(self.arena_size - addr);
            if let Some(s) = self.arena_range(addr, limit) {
                let n = s.iter().position(|&b| b == 0).unwrap_or(s.len());
                return Ok(s.get(..n).unwrap_or_default());
            }
        }
        Err(protection_fault(addr))
    }

    /// Port of `Memory::try_memmove`: move `len` bytes from `src` to `dst`
    /// inside the flat arena (overlap allowed). Returns false, moving
    /// nothing, when either range isn't in the arena or `dst` is below the
    /// read-only boundary.
    pub fn try_memmove(&mut self, dst: u64, src: u64, len: u64) -> bool {
        if dst == src || len == 0 {
            return true;
        }
        let dst_end = dst.wrapping_add(len);
        let src_end = src.wrapping_add(len);
        if dst >= self.initial_rodata_end
            && dst_end < self.arena_size
            && dst_end > dst
            && src_end < self.arena_size
            && src_end > src
        {
            // In range: the arena holds arena_size bytes (plus padding).
            if let (Ok(d), Ok(s), Ok(n)) = (
                usize::try_from(dst),
                usize::try_from(src),
                usize::try_from(len),
            ) {
                self.arena.copy_within(s..s + n, d);
                return true;
            }
        }
        false
    }

    /// Port of `Memory::memcmp(address_t, address_t, size_t)`: compare two
    /// guest ranges byte by byte (unsigned), faulting on unreadable memory.
    ///
    /// C++ returns `std::memcmp`'s `int` (or a byte difference on its slow
    /// path); only the sign is meaningful, so this returns an [`Ordering`].
    pub fn memcmp_guest(&self, p1: u64, p2: u64, len: u64) -> Result<Ordering> {
        Ok(self.memcmp_guest_diff(p1, p2, len)?.cmp(&0))
    }

    /// [`memcmp_guest`](Self::memcmp_guest) returning the difference of the
    /// first differing bytes (as unsigned bytes), or 0. C++ returns
    /// `std::memcmp`'s value, whose magnitude depends on the host's libc;
    /// this is what its slow path returns.
    pub(crate) fn memcmp_guest_diff(&self, p1: u64, p2: u64, len: u64) -> Result<i32> {
        fn diff(s1: &[u8], s2: &[u8]) -> i32 {
            s1.iter()
                .zip(s2)
                .find(|(a, b)| a != b)
                .map_or(0, |(&a, &b)| i32::from(a) - i32::from(b))
        }
        let e1 = p1.wrapping_add(len);
        let e2 = p2.wrapping_add(len);
        if e1 < p1 {
            return Err(protection_fault(p1));
        }
        if e2 < p2 {
            return Err(protection_fault(p2));
        }
        if e1.wrapping_sub(RWREAD_BEGIN) < self.read_boundary
            && e2.wrapping_sub(RWREAD_BEGIN) < self.read_boundary
            && p1 < e1
            && p2 < e2
            && let (Some(s1), Some(s2)) = (self.arena_range(p1, len), self.arena_range(p2, len))
        {
            return Ok(diff(s1, s2));
        }
        // Neither range crosses a page boundary.
        let pageno1 = p1 / PAGE_SIZE;
        let pageno2 = p2 / PAGE_SIZE;
        if pageno1 == e1.wrapping_sub(1) / PAGE_SIZE && pageno2 == e2.wrapping_sub(1) / PAGE_SIZE {
            let page1 = self.get_readable_pageno(pageno1)?;
            let page2 = self.get_readable_pageno(pageno2)?;
            let o1 = (p1 & PAGE_MASK) as usize;
            let o2 = (p2 & PAGE_MASK) as usize;
            let n = len as usize;
            let s1 = page1.get(o1..o1 + n).unwrap_or_default();
            let s2 = page2.get(o2..o2 + n).unwrap_or_default();
            return Ok(diff(s1, s2));
        }
        let (mut p1, mut p2, mut len) = (p1, p2, len);
        while len > 0 {
            let v1 = self.byte_at(p1)?;
            let v2 = self.byte_at(p2)?;
            if v1 != v2 {
                return Ok(i32::from(v1) - i32::from(v2));
            }
            p1 = p1.wrapping_add(1);
            p2 = p2.wrapping_add(1);
            len -= 1;
        }
        Ok(0)
    }

    /// Port of `Memory::memcmp(const void*, address_t, size_t)`: compare
    /// `data` with guest memory at `addr`, faulting on unreadable memory.
    /// C++'s slow path compares the host byte as a signed `char`, which
    /// flips the result for bytes above 0x7F; here all paths compare
    /// unsigned bytes, like its `std::memcmp` fast paths.
    pub fn memcmp(&self, data: &[u8], addr: u64) -> Result<Ordering> {
        let len = data.len() as u64;
        let end = addr.wrapping_add(len);
        if end < addr {
            return Err(protection_fault(addr));
        }
        if end.wrapping_sub(RWREAD_BEGIN) < self.read_boundary
            && addr < end
            && let Some(s2) = self.arena_range(addr, len)
        {
            return Ok(data.cmp(s2));
        }
        let pageno = addr / PAGE_SIZE;
        if pageno == end.wrapping_sub(1) / PAGE_SIZE {
            let page = self.get_readable_pageno(pageno)?;
            let off = (addr & PAGE_MASK) as usize;
            let s2 = page.get(off..off + data.len()).unwrap_or_default();
            return Ok(data.cmp(s2));
        }
        let mut p2 = addr;
        for &v1 in data {
            let v2 = self.byte_at(p2)?;
            if v1 != v2 {
                return Ok(v1.cmp(&v2));
            }
            p2 = p2.wrapping_add(1);
        }
        Ok(Ordering::Equal)
    }

    /// One readable byte, through the page table (the `memcmp` slow paths).
    fn byte_at(&self, addr: u64) -> Result<u8> {
        let page = self.get_readable_pageno(addr / PAGE_SIZE)?;
        Ok(page
            .get((addr & PAGE_MASK) as usize)
            .copied()
            .unwrap_or_default())
    }

    /// Port of `Memory::memcpy(address_t, Machine&, address_t, address_t)`:
    /// copy `len` bytes from another machine's memory at `src` to `dst`.
    /// Both ranges in the arena's write windows is one copy; otherwise C++'s
    /// loop of guest loads and stores (bytes until `src` is word-aligned,
    /// then words when both addresses share an alignment), with their
    /// faults. On a fault, the bytes before it have been copied.
    pub fn memcpy_from(&mut self, dst: u64, src_mem: &Memory, src: u64, len: u64) -> Result<()> {
        let (mut dst, mut src, mut len) = (dst, src, len);
        if let Some(s) = src_mem.try_memarray_mut_bytes(src, len) {
            if let Some(d) = self.try_memarray_mut_bytes_mut(dst, len) {
                d.copy_from_slice(s);
                return Ok(());
            }
            return self.memcpy(dst, s);
        }
        const W: u64 = 8;
        if (dst & (W - 1)) == (src & (W - 1)) {
            while (src & (W - 1)) != 0 && len > 0 {
                self.write_u8(dst, src_mem.read_u8(src)?)?;
                dst = dst.wrapping_add(1);
                src = src.wrapping_add(1);
                len -= 1;
            }
            while len >= 4 * W {
                if let Some(s) = src_mem.try_memarray_mut_bytes(src, len) {
                    return self.memcpy(dst, s);
                }
                for i in 0..4 {
                    let v = src_mem.read_u64(src.wrapping_add(i * W))?;
                    self.write_u64(dst.wrapping_add(i * W), v)?;
                }
                dst = dst.wrapping_add(4 * W);
                src = src.wrapping_add(4 * W);
                len -= 4 * W;
            }
            while len >= W {
                self.write_u64(dst, src_mem.read_u64(src)?)?;
                dst = dst.wrapping_add(W);
                src = src.wrapping_add(W);
                len -= W;
            }
        }
        while len > 0 {
            self.write_u8(dst, src_mem.read_u8(src)?)?;
            dst = dst.wrapping_add(1);
            src = src.wrapping_add(1);
            len -= 1;
        }
        Ok(())
    }

    /// Port of `Memory::memcpy(address_t, Machine&, address_t, address_t)`
    /// with this machine as the source (the native heap's `realloc` and the
    /// native `memcpy`): copy `len` guest bytes from `src` to `dst`, like
    /// [`memcpy_from`](Self::memcpy_from). Overlapping ranges in the arena
    /// are copied as if through a temporary buffer (C++'s `std::memcpy`
    /// leaves that undefined).
    pub fn memcpy_within(&mut self, dst: u64, src: u64, len: u64) -> Result<()> {
        let (mut dst, mut src, mut len) = (dst, src, len);
        if let Some(s) = self.write_window_range(src, len) {
            if let Some(d) = self.write_window_range(dst, len) {
                self.arena.copy_within(s, d.start);
                return Ok(());
            }
            let tmp = self.arena.get(s).unwrap_or_default().to_vec();
            return self.memcpy(dst, &tmp);
        }
        const W: u64 = 8;
        if (dst & (W - 1)) == (src & (W - 1)) {
            while (src & (W - 1)) != 0 && len > 0 {
                let v = self.read_u8(src)?;
                self.write_u8(dst, v)?;
                dst = dst.wrapping_add(1);
                src = src.wrapping_add(1);
                len -= 1;
            }
            while len >= 4 * W {
                if let Some(s) = self.write_window_range(src, len) {
                    let tmp = self.arena.get(s).unwrap_or_default().to_vec();
                    return self.memcpy(dst, &tmp);
                }
                for i in 0..4 {
                    let v = self.read_u64(src.wrapping_add(i * W))?;
                    self.write_u64(dst.wrapping_add(i * W), v)?;
                }
                dst = dst.wrapping_add(4 * W);
                src = src.wrapping_add(4 * W);
                len -= 4 * W;
            }
            while len >= W {
                let v = self.read_u64(src)?;
                self.write_u64(dst, v)?;
                dst = dst.wrapping_add(W);
                src = src.wrapping_add(W);
                len -= W;
            }
        }
        while len > 0 {
            let v = self.read_u8(src)?;
            self.write_u8(dst, v)?;
            dst = dst.wrapping_add(1);
            src = src.wrapping_add(1);
            len -= 1;
        }
        Ok(())
    }

    /// The arena indices of `[addr, addr + len)` if it is in the arena's
    /// write window (where C++ `try_memarray<uint8_t>` succeeds).
    fn write_window_range(&self, addr: u64, len: u64) -> Option<std::ops::Range<usize>> {
        if len == 0 {
            return None;
        }
        let offset = addr.wrapping_sub(self.initial_rodata_end);
        if offset < self.write_boundary && len <= self.write_boundary - offset {
            let a = usize::try_from(addr).ok()?;
            let n = usize::try_from(len).ok()?;
            let range = a..a.checked_add(n)?;
            return (range.end <= self.arena.len()).then_some(range);
        }
        None
    }

    /// C++ `try_memarray<uint8_t>` (non-const): the range in the arena's
    /// write window, read-only here.
    fn try_memarray_mut_bytes(&self, addr: u64, len: u64) -> Option<&[u8]> {
        if len == 0 {
            return None;
        }
        let offset = addr.wrapping_sub(self.initial_rodata_end);
        if offset < self.write_boundary && len <= self.write_boundary - offset {
            return self.arena_range(addr, len);
        }
        None
    }
    fn try_memarray_mut_bytes_mut(&mut self, addr: u64, len: u64) -> Option<&mut [u8]> {
        if len == 0 {
            return None;
        }
        let offset = addr.wrapping_sub(self.initial_rodata_end);
        if offset < self.write_boundary && len <= self.write_boundary - offset {
            let a = usize::try_from(addr).ok()?;
            let n = usize::try_from(len).ok()?;
            return self.arena.get_mut(a..a.checked_add(n)?);
        }
        None
    }

    /// Port of `Memory::memarray<const T>(addr, count, maxbytes)`: a
    /// read-only view of `count` elements at `addr`. A misaligned `addr`
    /// (unless `count` is 0), more than `maxbytes` (C++ default
    /// [`MEMVIEW_MAX`]) or memory that isn't one contiguous host range is a
    /// fault.
    pub fn memarray<T: GuestPod>(
        &self,
        addr: u64,
        count: u64,
        maxbytes: u64,
    ) -> Result<GuestSlice<'_, T>> {
        if count != 0 && misaligned::<T>(addr) {
            return Err(protection_fault(addr));
        }
        let len = array_bytes::<T>(addr, count, maxbytes)?;
        let view = self.memview_max(addr, len, maxbytes)?;
        if view.len() as u64 != len {
            return Err(protection_fault(addr));
        }
        Ok(GuestSlice::new(view))
    }

    /// Port of `Memory::memarray<T>(addr, count, maxbytes)` for a non-const
    /// `T`: like [`memarray`](Self::memarray), but writable (pages are made
    /// writable, faulting on read-only ones).
    pub fn memarray_mut<T: GuestPod>(
        &mut self,
        addr: u64,
        count: u64,
        maxbytes: u64,
    ) -> Result<GuestSliceMut<'_, T>> {
        if count != 0 && misaligned::<T>(addr) {
            return Err(protection_fault(addr));
        }
        let len = array_bytes::<T>(addr, count, maxbytes)?;
        let view = self.memview_mut_max(addr, len, maxbytes)?;
        if view.len() as u64 != len {
            return Err(protection_fault(addr));
        }
        Ok(GuestSliceMut::new(view))
    }

    /// Port of `Memory::try_memarray<const T>`: a read-only view of `count`
    /// elements if they are all in the arena's read window, else `None`
    /// (also for `count` 0). Too many bytes or a misaligned `addr` is
    /// still a fault.
    pub fn try_memarray<T: GuestPod>(
        &self,
        addr: u64,
        count: u64,
        maxbytes: u64,
    ) -> Result<Option<GuestSlice<'_, T>>> {
        if count == 0 {
            return Ok(None);
        }
        let len = array_bytes::<T>(addr, count, maxbytes)?;
        if misaligned::<T>(addr) {
            return Err(protection_fault(addr));
        }
        let offset = addr.wrapping_sub(RWREAD_BEGIN);
        if offset < self.read_boundary && len <= self.read_boundary - offset {
            return Ok(self.arena_range(addr, len).map(GuestSlice::new));
        }
        Ok(None)
    }

    /// Port of `Memory::try_memarray<T>` for a non-const `T`: a writable
    /// view if all elements are in the arena's write window, else `None`.
    pub fn try_memarray_mut<T: GuestPod>(
        &mut self,
        addr: u64,
        count: u64,
        maxbytes: u64,
    ) -> Result<Option<GuestSliceMut<'_, T>>> {
        if count == 0 {
            return Ok(None);
        }
        let len = array_bytes::<T>(addr, count, maxbytes)?;
        if misaligned::<T>(addr) {
            return Err(protection_fault(addr));
        }
        Ok(self
            .try_memarray_mut_bytes_mut(addr, len)
            .map(GuestSliceMut::new))
    }

    /// Port of `Memory::memspan<const T>`: like
    /// [`memarray`](Self::memarray), but a misaligned `addr` faults even
    /// for 0 elements.
    pub fn memspan<T: GuestPod>(
        &self,
        addr: u64,
        count: u64,
        maxlen: u64,
    ) -> Result<GuestSlice<'_, T>> {
        if misaligned::<T>(addr) {
            return Err(protection_fault(addr));
        }
        if count == 0 {
            return Ok(GuestSlice::new(&[]));
        }
        self.memarray(addr, count, maxlen)
    }

    /// Port of `Memory::memspan<T>` for a non-const `T`: like
    /// [`memarray_mut`](Self::memarray_mut), but a misaligned `addr` faults
    /// even for 0 elements.
    pub fn memspan_mut<T: GuestPod>(
        &mut self,
        addr: u64,
        count: u64,
        maxlen: u64,
    ) -> Result<GuestSliceMut<'_, T>> {
        if misaligned::<T>(addr) {
            return Err(protection_fault(addr));
        }
        if count == 0 {
            return Ok(GuestSliceMut::new(&mut []));
        }
        self.memarray_mut(addr, count, maxlen)
    }

    /// Read one [`GuestPod`] from guest memory (C++ `memcpy_out(&value,
    /// addr, sizeof(T))`), with no alignment requirement.
    pub fn read_value<T: GuestPod>(&self, addr: u64) -> Result<T> {
        let mut buf = vec![0u8; T::SIZE];
        self.memcpy_out(&mut buf, addr)?;
        Ok(T::read_guest(&buf))
    }

    /// Write one [`GuestPod`] to guest memory (C++ `memcpy(addr, &value,
    /// sizeof(T))`), with no alignment requirement.
    pub fn write_value<T: GuestPod>(&mut self, addr: u64, value: &T) -> Result<()> {
        let mut buf = Vec::with_capacity(T::SIZE);
        value.write_guest(&mut buf);
        self.memcpy(addr, &buf)
    }
}
