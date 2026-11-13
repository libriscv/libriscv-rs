//! Port of `guest/guest_cpp_vector.hpp`: libstdc++'s (and libc++'s)
//! `std::vector`.

use std::marker::PhantomData;

use super::{
    GuestInit, GuestPtr, GuestStdString, GuestType, MAX_BYTES, container_error, guest_alloc,
    guest_free, pass_by_value,
};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::vmcall::{ArgRegs, VmCallArg};

/// `std::vector<T>` (C++ `GuestStdVector`): begin, end and capacity
/// pointers. The elements live in a native heap allocation of their own.
///
/// Copies are shallow, like the C++ mirror's: a copy shares the elements
/// with the original, and only one of them may free them.
pub struct GuestStdVector<T> {
    pub ptr_begin: u64,
    pub ptr_end: u64,
    pub ptr_capacity: u64,
    _type: PhantomData<fn() -> T>,
}

impl<T> GuestStdVector<T> {
    pub const fn from_raw(ptr_begin: u64, ptr_end: u64, ptr_capacity: u64) -> Self {
        Self {
            ptr_begin,
            ptr_end,
            ptr_capacity,
            _type: PhantomData,
        }
    }
}

impl<T> Default for GuestStdVector<T> {
    fn default() -> Self {
        Self::from_raw(0, 0, 0)
    }
}
impl<T> Clone for GuestStdVector<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GuestStdVector<T> {}
impl<T> PartialEq for GuestStdVector<T> {
    fn eq(&self, o: &Self) -> bool {
        (self.ptr_begin, self.ptr_end, self.ptr_capacity)
            == (o.ptr_begin, o.ptr_end, o.ptr_capacity)
    }
}
impl<T> Eq for GuestStdVector<T> {}
impl<T> std::fmt::Debug for GuestStdVector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestStdVector")
            .field("ptr_begin", &self.ptr_begin)
            .field("ptr_end", &self.ptr_end)
            .field("ptr_capacity", &self.ptr_capacity)
            .finish()
    }
}

impl<T> GuestPod for GuestStdVector<T> {
    const SIZE: usize = 24;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.ptr_begin.to_le_bytes());
        out.extend_from_slice(&self.ptr_end.to_le_bytes());
        out.extend_from_slice(&self.ptr_capacity.to_le_bytes());
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self::from_raw(
            u64::read_guest(bytes),
            u64::read_guest(bytes.get(8..).unwrap_or_default()),
            u64::read_guest(bytes.get(16..).unwrap_or_default()),
        )
    }
}

impl<T: GuestType> GuestStdVector<T> {
    /// `sizeof(T)`, never zero.
    const ELEM: u64 = if T::SIZE == 0 { 1 } else { T::SIZE as u64 };

    pub const fn data(&self) -> u64 {
        self.ptr_begin
    }
    pub const fn size_bytes(&self) -> u64 {
        self.ptr_end.wrapping_sub(self.ptr_begin)
    }
    pub const fn capacity_bytes(&self) -> u64 {
        self.ptr_capacity.wrapping_sub(self.ptr_begin)
    }
    pub const fn size(&self) -> usize {
        (self.size_bytes() / Self::ELEM) as usize
    }
    pub const fn capacity(&self) -> usize {
        (self.capacity_bytes() / Self::ELEM) as usize
    }
    pub const fn empty(&self) -> bool {
        self.size() == 0
    }

    /// The guest address of element `index` (C++ `address_at`).
    pub fn address_at(&self, index: usize) -> Result<u64> {
        if index >= self.size() {
            return Err(container_error("Guest std::vector index out of range"));
        }
        Ok(self
            .ptr_begin
            .wrapping_add((index as u64).wrapping_mul(Self::ELEM)))
    }

    /// Element `index` (C++ `at`, with `as_array`'s size limit).
    pub fn at(&self, index: usize) -> Result<GuestPtr<T>> {
        let addr = self.address_at(index)?;
        if self.size_bytes() > MAX_BYTES as u64 {
            return Err(container_error("Guest std::vector has size > max_bytes"));
        }
        Ok(GuestPtr::new(addr))
    }

    /// Every element (C++ `begin(machine)`..`end(machine)`).
    pub fn iter(&self) -> Result<impl Iterator<Item = GuestPtr<T>> + use<T>> {
        if self.size_bytes() > MAX_BYTES as u64 {
            return Err(container_error("Guest std::vector has size > max_bytes"));
        }
        let (begin, n) = (self.ptr_begin, self.size() as u64);
        Ok((0..n).map(move |i| GuestPtr::new(begin.wrapping_add(i * Self::ELEM))))
    }

    /// Copy the elements out (C++ `to_vector`): shallow copies.
    pub fn to_vec(&self, m: &Machine) -> Result<Vec<T>> {
        if self.size_bytes() > self.capacity_bytes() {
            return Err(container_error("Guest std::vector has size > capacity"));
        }
        self.iter()?.map(|p| p.get(m)).collect()
    }
}

impl GuestStdVector<GuestStdString> {
    /// The strings as host strings (C++ `to_string_vector`).
    pub fn to_string_vector(&self, m: &Machine) -> Result<Vec<String>> {
        self.iter()?
            .map(|p| p.get(m).and_then(|s| s.to_string(m)))
            .collect()
    }
}

impl<T: GuestType> GuestType for GuestStdVector<T> {
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).free_elements_and_array(m)
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        if self.ptr_begin != 0 {
            for p in self.iter()? {
                p.free(m)?;
            }
            guest_free(m, self.ptr_begin)?;
            *self = Self::default();
        }
        Ok(())
    }
}

impl<T: GuestType> GuestStdVector<T> {
    /// A host-side vector holding `values`, each constructed in a new
    /// native heap allocation (C++ `GuestStdVector(machine, vec)`; the
    /// vector itself needs no address).
    pub fn from_values<A: GuestInit<T>>(m: &mut Machine, values: &[A]) -> Result<Self> {
        if values.is_empty() {
            return Ok(Self::default());
        }
        let bytes = (values.len() as u64)
            .checked_mul(Self::ELEM)
            .and_then(|b| usize::try_from(b).ok())
            .ok_or(container_error("Guest std::vector: too many elements"))?;
        let begin = guest_alloc(m, bytes)?;
        for (i, value) in values.iter().enumerate() {
            value.init_at(m, begin.wrapping_add(i as u64 * Self::ELEM))?;
        }
        let end = begin.wrapping_add(bytes as u64);
        Ok(Self::from_raw(begin, end, end))
    }
}

/// Construct from host values (C++ `GuestStdVector(machine, vec)`): an
/// empty slice is an empty vector without an allocation, otherwise each
/// element is constructed at its own address.
impl<T: GuestType, A: GuestInit<T>> GuestInit<GuestStdVector<T>> for [A] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<GuestStdVector<T>>::new(addr);
        ptr.set(m, &GuestStdVector::default())?;
        if self.is_empty() {
            return Ok(());
        }
        ptr.construct_all(m, self)
    }
    /// C++ `assign(machine, vec)`: allocates even for no elements.
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<GuestStdVector<T>>::new(addr).construct_all(m, self)
    }
}
impl<T: GuestType, A: GuestInit<T>> GuestInit<GuestStdVector<T>> for Vec<A> {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        self.as_slice().init_at(m, addr)
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        self.as_slice().assign_at(m, addr)
    }
}
impl<T: GuestType, A: GuestInit<T>, const N: usize> GuestInit<GuestStdVector<T>> for [A; N] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        self.as_slice().init_at(m, addr)
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        self.as_slice().assign_at(m, addr)
    }
}

impl<T: GuestType> GuestPtr<GuestStdVector<T>> {
    const ELEM: u64 = GuestStdVector::<T>::ELEM;

    pub fn size(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.size())
    }
    pub fn capacity(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.capacity())
    }
    pub fn empty(&self, m: &Machine) -> Result<bool> {
        Ok(self.get(m)?.empty())
    }
    /// Element `index` (C++ `at`).
    pub fn at(&self, m: &Machine, index: usize) -> Result<GuestPtr<T>> {
        self.get(m)?.at(index)
    }
    /// The guest address of element `index` (C++ `address_at`).
    pub fn address_at(&self, m: &Machine, index: usize) -> Result<u64> {
        self.get(m)?.address_at(index)
    }
    /// Copy the elements out (C++ `to_vector`).
    pub fn to_vec(&self, m: &Machine) -> Result<Vec<T>> {
        self.get(m)?.to_vec(m)
    }

    /// C++ `alloc`: free the vector, then allocate room for `elements`.
    fn alloc(&self, m: &mut Machine, elements: usize) -> Result<GuestStdVector<T>> {
        self.free_elements_and_array(m)?;
        let bytes = (elements as u64)
            .checked_mul(Self::ELEM)
            .and_then(|b| usize::try_from(b).ok())
            .ok_or(container_error("Guest std::vector: too many elements"))?;
        let begin = guest_alloc(m, bytes)?;
        let v = GuestStdVector::from_raw(begin, begin, begin.wrapping_add(bytes as u64));
        self.set(m, &v)?;
        Ok(v)
    }

    /// Allocate exactly `values.len()` elements and construct them; the
    /// end moves only after all of them are constructed.
    fn construct_all<A: GuestInit<T>>(&self, m: &mut Machine, values: &[A]) -> Result<()> {
        let mut v = self.alloc(m, values.len())?;
        for (i, value) in values.iter().enumerate() {
            value.init_at(m, v.ptr_begin.wrapping_add(i as u64 * Self::ELEM))?;
        }
        v.ptr_end = v.ptr_begin.wrapping_add(values.len() as u64 * Self::ELEM);
        self.set(m, &v)
    }

    /// Append one element constructed from `value` (C++ `push_back`, for a
    /// `T`, a string or a host vector).
    pub fn push_back<A: GuestInit<T> + ?Sized>(&self, m: &mut Machine, value: &A) -> Result<()> {
        let mut v = self.get(m)?;
        if v.size_bytes() >= v.capacity_bytes() {
            // C++ increase_capacity
            self.reserve(m, v.capacity().wrapping_mul(2).wrapping_add(4))?;
            v = self.get(m)?;
        }
        value.init_at(m, v.ptr_end)?;
        v.ptr_end = v.ptr_end.wrapping_add(Self::ELEM);
        self.set(m, &v)
    }

    /// Remove and free the last element (C++ `pop_back`).
    pub fn pop_back(&self, m: &mut Machine) -> Result<()> {
        let mut v = self.get(m)?;
        if v.size() == 0 {
            return Err(container_error("Guest std::vector is empty"));
        }
        T::free_at(m, v.address_at(v.size() - 1)?)?;
        v.ptr_end = v.ptr_end.wrapping_sub(Self::ELEM);
        self.set(m, &v)
    }

    /// Append shallow copies of `values` (C++ `append`).
    pub fn append(&self, m: &mut Machine, values: &[T]) -> Result<()> {
        let v = self.get(m)?;
        let count = values.len() as u64;
        if v.size_bytes().wrapping_add(count.wrapping_mul(Self::ELEM)) > v.capacity_bytes() {
            self.reserve(m, v.size().wrapping_add(values.len()))?;
        }
        let mut v = self.get(m)?;
        for value in values {
            value.init_at(m, v.ptr_end)?;
            v.ptr_end = v.ptr_end.wrapping_add(Self::ELEM);
        }
        self.set(m, &v)
    }

    /// Free every element, keeping the allocation (C++ `clear`).
    pub fn clear(&self, m: &mut Machine) -> Result<()> {
        let mut v = self.get(m)?;
        for p in v.iter()? {
            p.free(m)?;
        }
        v.ptr_end = v.ptr_begin;
        self.set(m, &v)
    }

    /// Take over `count` initialized elements at `array`, which must have
    /// been allocated on the native heap (C++ `assume_ownership`).
    pub fn assume_ownership(&self, m: &mut Machine, array: u64, count: usize) -> Result<()> {
        self.free_elements_and_array(m)?;
        let end = array.wrapping_add((count as u64).wrapping_mul(Self::ELEM));
        self.set(m, &GuestStdVector::from_raw(array, end, end))
    }

    /// Grow or shrink to `new_size` elements, default-constructing new
    /// ones (C++ `resize`).
    pub fn resize(&self, m: &mut Machine, new_size: usize) -> Result<()>
    where
        T: Default,
    {
        let v = self.get(m)?;
        let size = v.size();
        if new_size < size {
            for i in new_size..size {
                T::free_at(m, v.address_at(i)?)?;
            }
        } else if new_size > size {
            if new_size > v.capacity() {
                self.reserve(m, new_size)?;
            }
            let v = self.get(m)?;
            for i in size..new_size {
                T::default().init_at(m, v.ptr_begin.wrapping_add(i as u64 * Self::ELEM))?;
            }
        }
        let mut v = self.get(m)?;
        v.ptr_end = v
            .ptr_begin
            .wrapping_add((new_size as u64).wrapping_mul(Self::ELEM));
        self.set(m, &v)
    }

    /// Make room for `elements` (C++ `reserve`): the elements move to a new
    /// allocation, and the ones that point into themselves are told.
    pub fn reserve(&self, m: &mut Machine, elements: usize) -> Result<()> {
        let old = self.get(m)?;
        if elements <= old.capacity() {
            return Ok(());
        }
        // Move out first, so that alloc() has nothing to free.
        self.set(m, &GuestStdVector::default())?;
        let mut v = self.alloc(m, elements)?;
        if !old.empty() {
            if old.size_bytes() > MAX_BYTES as u64 {
                return Err(container_error("Guest std::vector has size > max_bytes"));
            }
            let mut bytes = vec![0; old.size_bytes() as usize];
            m.memory.memcpy_out(&mut bytes, old.ptr_begin)?;
            m.memory.memcpy(v.ptr_begin, &bytes)?;
            // Free the old array only (the elements moved).
            guest_free(m, old.ptr_begin)?;
        }
        v.ptr_end = v.ptr_begin.wrapping_add(old.size_bytes());
        self.set(m, &v)?;
        if T::SELF_REFERENCING {
            for p in v.iter()? {
                p.relocate(m)?;
            }
        }
        Ok(())
    }

    /// Free the elements and the allocation (C++ `free`).
    fn free_elements_and_array(&self, m: &mut Machine) -> Result<()> {
        let v = self.get(m)?;
        if v.ptr_begin != 0 {
            for p in v.iter()? {
                p.free(m)?;
            }
            guest_free(m, v.ptr_begin)?;
            self.set(m, &GuestStdVector::default())?;
        }
        Ok(())
    }
}

impl GuestPtr<GuestStdVector<GuestStdString>> {
    /// The strings as host strings (C++ `to_string_vector`).
    pub fn to_string_vector(&self, m: &Machine) -> Result<Vec<String>> {
        self.get(m)?.to_string_vector(m)
    }
}

/// By value: pushed onto the stack (C++ `setup_call`).
impl<T: GuestType> VmCallArg for GuestStdVector<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        pass_by_value(self, m, regs)
    }
}
