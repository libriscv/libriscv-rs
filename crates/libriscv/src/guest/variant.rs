//! Port of `guest/guest_cpp_variant.hpp`: libstdc++'s (and libc++'s)
//! `std::variant`.
//!
//! The alternatives are a Rust enum declared with [`guest_variant!`],
//! which implements [`GuestVariant`] for it and [`VariantAlternative`] for
//! each alternative type: the enum is what C++ calls `visit`, and the
//! alternative types give `index_of<T>`, `get<T>`, `get_if<T>`,
//! `holds_alternative<T>` and `emplace<T>`.
//!
//! [`guest_variant!`]: crate::guest_variant

use super::{GuestInit, GuestPtr, GuestType, container_error, pass_by_value, round_up};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::vmcall::{ArgRegs, VmCallArg};

/// The alternatives of a [`GuestStdVariant`]: implemented by
/// [`guest_variant!`](crate::guest_variant) for an enum with one guest
/// type per variant, in the order of the C++ `std::variant`.
pub trait GuestVariant: Sized + Clone {
    /// The number of alternatives.
    const COUNT: usize;
    /// The alignment of the union of the alternatives.
    const UNION_ALIGN: usize;
    /// The size of the union, rounded up to its alignment.
    const UNION_SIZE: usize;
    /// True when an alternative points back into itself.
    const SELF_REFERENCING: bool;
    /// The index of the active alternative.
    fn index(&self) -> usize;
    /// Append the active alternative's guest bytes.
    fn write_alternative(&self, out: &mut Vec<u8>);
    /// Decode alternative `index` from the union's bytes.
    fn read_alternative(index: usize, bytes: &[u8]) -> Option<Self>;
    /// Free alternative `index`, living at `addr`.
    fn free_alternative(index: usize, m: &mut Machine, addr: u64) -> Result<()>;
    /// Relocate alternative `index`, now living at `addr`.
    fn relocate_alternative(index: usize, m: &mut Machine, addr: u64) -> Result<()>;
    /// Free the active alternative of this host-side value.
    fn free_value(&mut self, m: &mut Machine) -> Result<()>;
}

/// One alternative type of the variant `E` (implemented by
/// [`guest_variant!`](crate::guest_variant)).
pub trait VariantAlternative<E>: GuestType {
    /// The alternative's index (C++ `index_of<T>`).
    const INDEX: usize;
    fn wrap(self) -> E;
    fn get(e: &E) -> Option<&Self>;
    fn get_mut(e: &mut E) -> Option<&mut Self>;
}

#[doc(hidden)]
pub const fn max_of(values: &[usize]) -> usize {
    let mut max = 0;
    let mut i = 0;
    while i < values.len() {
        // In bounds: i < values.len().
        #[allow(clippy::indexing_slicing)]
        if values[i] > max {
            max = values[i];
        }
        i += 1;
    }
    max
}

/// Declare an enum mirroring a guest `std::variant<Types...>`: one tuple
/// variant per alternative, in the guest's order, each holding a guest
/// type ([`GuestType`]). Alternative types must be distinct (as for C++
/// `set`/`get<T>`).
///
/// ```
/// libriscv::guest_variant! {
///     #[derive(Debug, Clone, PartialEq)]
///     pub enum Value {
///         Bool(bool),
///         Int(i32),
///         Double(f64),
///         Str(libriscv::guest::GuestStdString),
///     }
/// }
/// type CppValue = libriscv::guest::GuestStdVariant<Value>;
/// assert_eq!(<CppValue as libriscv::GuestPod>::SIZE, 40);
/// ```
#[macro_export]
macro_rules! guest_variant {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident($ty:ty)),+ $(,)? }) => {
        $(#[$meta])*
        $vis enum $name {
            $($variant($ty)),+
        }

        impl $crate::guest::GuestVariant for $name {
            const COUNT: usize = [$(stringify!($variant)),+].len();
            const UNION_ALIGN: usize = $crate::guest::max_of(
                &[$(<$ty as $crate::GuestPod>::ALIGN),+]);
            const UNION_SIZE: usize = $crate::guest::max_of(
                &[$(<$ty as $crate::GuestPod>::SIZE),+])
                .div_ceil(<Self as $crate::guest::GuestVariant>::UNION_ALIGN)
                * <Self as $crate::guest::GuestVariant>::UNION_ALIGN;
            const SELF_REFERENCING: bool =
                false $(|| <$ty as $crate::guest::GuestType>::SELF_REFERENCING)+;
            fn index(&self) -> usize {
                let mut _i = 0usize;
                $(
                    if let Self::$variant(_) = self {
                        return _i;
                    }
                    _i += 1;
                )+
                usize::MAX
            }
            fn write_alternative(&self, out: &mut Vec<u8>) {
                match self {
                    $(Self::$variant(v) => $crate::GuestPod::write_guest(v, out),)+
                }
            }
            fn read_alternative(index: usize, bytes: &[u8]) -> Option<Self> {
                let mut _i = 0usize;
                $(
                    if index == _i {
                        return Some(Self::$variant(
                            <$ty as $crate::GuestPod>::read_guest(bytes)));
                    }
                    _i += 1;
                )+
                None
            }
            fn free_alternative(index: usize, m: &mut $crate::Machine, addr: u64)
                -> $crate::Result<()>
            {
                let mut _i = 0usize;
                $(
                    if index == _i {
                        return <$ty as $crate::guest::GuestType>::free_at(m, addr);
                    }
                    _i += 1;
                )+
                Ok(())
            }
            fn free_value(&mut self, m: &mut $crate::Machine) -> $crate::Result<()> {
                match self {
                    $(Self::$variant(v) => $crate::guest::GuestType::free_value(v, m),)+
                }
            }
            fn relocate_alternative(index: usize, m: &mut $crate::Machine, addr: u64)
                -> $crate::Result<()>
            {
                let mut _i = 0usize;
                $(
                    if index == _i {
                        return <$ty as $crate::guest::GuestType>::relocate_at(m, addr);
                    }
                    _i += 1;
                )+
                Ok(())
            }
        }

        $crate::guest_variant!(@alternatives $name; 0usize; $($variant($ty)),+);
    };
    (@alternatives $name:ident; $index:expr; $variant:ident($ty:ty) $(, $rest:ident($rty:ty))*) => {
        impl $crate::guest::VariantAlternative<$name> for $ty {
            const INDEX: usize = $index;
            fn wrap(self) -> $name {
                $name::$variant(self)
            }
            #[allow(unreachable_patterns)]
            fn get(e: &$name) -> Option<&Self> {
                match e {
                    $name::$variant(v) => Some(v),
                    _ => None,
                }
            }
            #[allow(unreachable_patterns)]
            fn get_mut(e: &mut $name) -> Option<&mut Self> {
                match e {
                    $name::$variant(v) => Some(v),
                    _ => None,
                }
            }
        }
        $crate::guest_variant!(@alternatives $name; ($index) + 1; $($rest($rty)),*);
    };
    (@alternatives $name:ident; $index:expr; ) => {};
}

/// `std::variant` (C++ `GuestStdVariant`): the union of the alternatives,
/// followed by the index of the active one (one byte for up to 255
/// alternatives). `None` is valueless (`std::variant_npos`).
///
/// Copies are shallow, like the C++ mirror's: guest memory owned by the
/// active alternative is shared with the copy.
#[derive(Debug, Clone, PartialEq)]
pub struct GuestStdVariant<E> {
    value: Option<E>,
}

impl<E: GuestVariant> GuestStdVariant<E> {
    const INDEX_SIZE: usize = if E::COUNT <= 255 { 1 } else { 2 };
    /// The index of a valueless variant, as stored.
    const NPOS: u64 = if E::COUNT <= 255 { 0xFF } else { 0xFFFF };

    pub const fn new(value: E) -> Self {
        Self { value: Some(value) }
    }
    /// A variant that holds nothing (valueless by exception).
    pub const fn valueless() -> Self {
        Self { value: None }
    }
    /// The active alternative, or `None` when valueless.
    pub const fn value(&self) -> Option<&E> {
        self.value.as_ref()
    }
    pub const fn value_mut(&mut self) -> Option<&mut E> {
        self.value.as_mut()
    }
    pub fn into_inner(self) -> Option<E> {
        self.value
    }
    /// The index of the active alternative, or `usize::MAX`
    /// (`std::variant_npos`).
    pub fn index(&self) -> usize {
        self.value.as_ref().map_or(usize::MAX, E::index)
    }
    pub const fn valueless_by_exception(&self) -> bool {
        self.value.is_none()
    }
    /// The index of the alternative `T` (C++ `index_of<T>`).
    pub const fn index_of<T: VariantAlternative<E>>() -> usize {
        T::INDEX
    }
    pub fn holds_alternative<T: VariantAlternative<E>>(&self) -> bool {
        self.get_if::<T>().is_some()
    }
    /// The alternative `T`, or `None` (C++ `get_if<T>`).
    pub fn get_if<T: VariantAlternative<E>>(&self) -> Option<&T> {
        self.value.as_ref().and_then(T::get)
    }
    /// The alternative `T`, or an error (C++ `get<T>`).
    pub fn get<T: VariantAlternative<E>>(&self) -> Result<&T> {
        self.get_if::<T>().ok_or(container_error(
            "Guest std::variant does not hold the given alternative",
        ))
    }
}

impl<E: GuestVariant> Default for GuestStdVariant<E> {
    /// A zeroed first alternative, like a default-constructed `std::variant`.
    fn default() -> Self {
        Self {
            value: E::read_alternative(0, &vec![0; E::UNION_SIZE]),
        }
    }
}

impl<E: GuestVariant> From<E> for GuestStdVariant<E> {
    fn from(value: E) -> Self {
        Self::new(value)
    }
}

impl<E: GuestVariant> GuestPod for GuestStdVariant<E> {
    const ALIGN: usize = if E::UNION_ALIGN > Self::INDEX_SIZE {
        E::UNION_ALIGN
    } else {
        Self::INDEX_SIZE
    };
    const SIZE: usize = round_up(E::UNION_SIZE + Self::INDEX_SIZE, Self::ALIGN);
    fn write_guest(&self, out: &mut Vec<u8>) {
        let start = out.len();
        if let Some(v) = &self.value {
            v.write_alternative(out);
        }
        out.resize(start + E::UNION_SIZE, 0);
        let index = self.value.as_ref().map_or(Self::NPOS, |v| v.index() as u64);
        out.extend_from_slice(
            index
                .to_le_bytes()
                .get(..Self::INDEX_SIZE)
                .unwrap_or_default(),
        );
        out.resize(start + Self::SIZE, 0);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        let at = |i: usize| u64::from(bytes.get(E::UNION_SIZE + i).copied().unwrap_or(0xFF));
        let index = if Self::INDEX_SIZE == 1 {
            at(0)
        } else {
            at(0) | (at(1) << 8)
        };
        let value = usize::try_from(index)
            .ok()
            .filter(|&i| i < E::COUNT)
            .and_then(|i| E::read_alternative(i, bytes));
        Self { value }
    }
}

impl<E: GuestVariant> GuestType for GuestStdVariant<E> {
    const SELF_REFERENCING: bool = E::SELF_REFERENCING;
    /// Destroy the active alternative of this host-side value, leaving it
    /// valueless (C++ `free`).
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        if let Some(v) = &mut self.value {
            v.free_value(m)?;
        }
        self.value = None;
        Ok(())
    }
    /// Destroy the active alternative, leaving the variant valueless.
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<Self>::new(addr);
        if let Some(index) = ptr.active_index(m)? {
            E::free_alternative(index, m, addr)?;
        }
        ptr.set_index(m, Self::NPOS)
    }
    fn relocate_at(m: &mut Machine, addr: u64) -> Result<()> {
        match GuestPtr::<Self>::new(addr).active_index(m)? {
            Some(index) => E::relocate_alternative(index, m, addr),
            None => Ok(()),
        }
    }
}

/// Construct the variant holding `self` (a shallow copy of the
/// alternative, relocated: C++ `set` with a value of an alternative type).
impl<E: GuestVariant> GuestInit<GuestStdVariant<E>> for E {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestStdVariant::new(self.clone()).init_at(m, addr)
    }
}

impl<E: GuestVariant> GuestPtr<GuestStdVariant<E>> {
    fn index_addr(&self) -> u64 {
        self.address().wrapping_add(E::UNION_SIZE as u64)
    }
    fn raw_index(&self, m: &Machine) -> Result<u64> {
        if GuestStdVariant::<E>::INDEX_SIZE == 1 {
            Ok(u64::from(m.memory.read_u8(self.index_addr())?))
        } else {
            Ok(u64::from(m.memory.read_u16(self.index_addr())?))
        }
    }
    fn set_index(&self, m: &mut Machine, index: u64) -> Result<()> {
        if GuestStdVariant::<E>::INDEX_SIZE == 1 {
            m.memory.write_u8(self.index_addr(), index as u8)
        } else {
            m.memory.write_u16(self.index_addr(), index as u16)
        }
    }
    fn active_index(&self, m: &Machine) -> Result<Option<usize>> {
        Ok(usize::try_from(self.raw_index(m)?)
            .ok()
            .filter(|&i| i < E::COUNT))
    }

    /// The index of the active alternative, or `usize::MAX`.
    pub fn index(&self, m: &Machine) -> Result<usize> {
        Ok(self.active_index(m)?.unwrap_or(usize::MAX))
    }
    pub fn valueless_by_exception(&self, m: &Machine) -> Result<bool> {
        Ok(self.active_index(m)?.is_none())
    }
    pub fn holds_alternative<T: VariantAlternative<E>>(&self, m: &Machine) -> Result<bool> {
        Ok(self.active_index(m)? == Some(T::INDEX))
    }
    /// The alternative `T`, in place (C++ `get<T>`, a reference into guest
    /// memory), or an error when it isn't active.
    pub fn alternative<T: VariantAlternative<E>>(&self, m: &Machine) -> Result<GuestPtr<T>> {
        if !self.holds_alternative::<T>(m)? {
            return Err(container_error(
                "Guest std::variant does not hold the given alternative",
            ));
        }
        Ok(self.cast())
    }
    /// Replace the value with an alternative `T` constructed from a host
    /// value (C++ `emplace<T>`, and `set` with a string, vector or map).
    pub fn emplace<T, A>(&self, m: &mut Machine, value: &A) -> Result<GuestPtr<T>>
    where
        T: VariantAlternative<E>,
        A: GuestInit<T> + ?Sized,
    {
        <GuestStdVariant<E> as GuestType>::free_at(m, self.address())?;
        value.init_at(m, self.address())?;
        self.set_index(m, T::INDEX as u64)?;
        Ok(self.cast())
    }
}

/// By value: pushed onto the stack, and relocated there (C++ `setup_call`).
impl<E: GuestVariant> VmCallArg for GuestStdVariant<E> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        pass_by_value(self, m, regs)
    }
}
