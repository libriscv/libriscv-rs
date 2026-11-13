//! Port of `guest_datatypes.hpp`, `guest/guest_common.hpp`,
//! `guest/guest_cpp_common.hpp` and `guest/guest_arena_object.hpp`:
//! host-side access to the C++ standard containers that live in guest
//! memory, in libstdc++'s layout (`std::string`, `std::vector`,
//! `std::unordered_map`, `std::variant`).
//!
//! # Design
//!
//! C++ views these containers through raw pointers into guest memory and
//! changes them in place; a `std::string` (small-string optimization) and a
//! `std::unordered_map` (its `before_begin` node and single bucket) point
//! back into themselves. Here:
//!
//! * Each container is a [`GuestPod`] value type with the guest layout
//!   ([`GuestStdString`], [`GuestStdVector`], [`GuestStdUnorderedMap`],
//!   [`GuestStdVariant`]), so it can be read, copied, nested and passed by
//!   value like the C++ mirror. Read-only methods work on such values.
//! * Everything that changes a container works *in place*, on the object
//!   at its guest address, through a typed handle: [`GuestPtr<T>`] (C++
//!   `GuestRef`, or a `T*` from `memarray`). That keeps the self-pointers
//!   coherent without a host copy that could go stale. Methods take the
//!   machine explicitly, so handles don't borrow it and can be passed to
//!   `vmcall` while alive.
//! * [`ScopedArenaObject<T>`] owns a `T` on the native heap (the machine's
//!   arena, see `setup_native_heap`). C++ frees it in a destructor; Rust
//!   can't reach the machine from `Drop`, so call
//!   [`ScopedArenaObject::free`] (it is `#[must_use]`).
//! * The traits replace C++'s type-trait dispatch: [`GuestType`] (free and
//!   relocate an object at an address: `free_guest_object`,
//!   `relocate_guest_object`) and [`GuestInit`] (construct one from a host
//!   value: `construct_guest_object`).
//!
//! Only RV64 guests: the 32-bit variants of the layouts and of `std::hash`
//! are not ported. The Rust guest containers (`guest_rust_*.hpp`) are not
//! ported yet.

mod hash;
mod string;
mod unordered_map;
mod variant;
mod vector;

pub use hash::{
    CustomHash, GuestHashKey, GuestKeyArg, HashPolicy, SEED as HASH_SEED, StdHash, hash_bytes,
};
pub use string::GuestStdString;
pub use unordered_map::GuestStdUnorderedMap;
#[doc(hidden)]
pub use variant::max_of;
pub use variant::{GuestStdVariant, GuestVariant, VariantAlternative};
pub use vector::GuestStdVector;

use std::marker::PhantomData;

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::vmcall::{ArgRegs, VmCallArg};

/// The default size limit when reading a container's contents (C++ `16UL
/// << 20` everywhere).
pub const MAX_BYTES: usize = 16 << 20;

pub(crate) const fn round_up(value: usize, align: usize) -> usize {
    if align <= 1 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

/// A C++ `std::runtime_error`/`std::out_of_range` from the containers.
pub(crate) const fn container_error(message: &'static str) -> MachineError {
    MachineError::new(ExceptionKind::IllegalOperation, message, 0)
}

/// Allocate from the native heap; C++ throws `std::bad_alloc` on a null
/// result (where it checks).
pub(crate) fn guest_alloc(m: &mut Machine, size: usize) -> Result<u64> {
    let addr = arena_malloc(m, size)?;
    if addr == 0 {
        return Err(MachineError::new(
            ExceptionKind::OutOfMemory,
            "Guest container: out of native heap memory",
            size as u64,
        ));
    }
    Ok(addr)
}

/// Free to the native heap. Like C++, the containers ignore the result.
pub(crate) fn guest_free(m: &mut Machine, addr: u64) -> Result<()> {
    arena_free(m, addr)
}

/// C++ `machine.arena().malloc(size)`.
fn arena_malloc(m: &mut Machine, size: usize) -> Result<u64> {
    Ok(u64::from(m.arena_mut()?.malloc(size as u64)?))
}
/// C++ `machine.arena().free(addr)`: the arena's pointers are 32-bit,
/// and C++ truncates the address implicitly.
fn arena_free(m: &mut Machine, addr: u64) -> Result<()> {
    #[allow(clippy::cast_possible_truncation)]
    let _ = m.arena_mut()?.free(addr as u32);
    Ok(())
}

pub(crate) fn read_u64(m: &Machine, addr: u64) -> Result<u64> {
    m.memory.read_u64(addr)
}

pub(crate) fn write_u64(m: &mut Machine, addr: u64, value: u64) -> Result<()> {
    m.memory.write_u64(addr, value)
}

/// A typed guest address (C++ `GuestRef<T>`, or the `T*` that `memarray`
/// returns): the handle through which a guest object is read, written and
/// changed in place. It doesn't borrow the machine; every access checks
/// the memory, and faults like C++ `memarray` (including on misalignment).
pub struct GuestPtr<T> {
    addr: u64,
    _type: PhantomData<fn() -> T>,
}

impl<T> GuestPtr<T> {
    pub const fn new(addr: u64) -> Self {
        Self {
            addr,
            _type: PhantomData,
        }
    }
    /// The guest address.
    pub const fn address(&self) -> u64 {
        self.addr
    }
    /// The same address, viewed as another type (C++
    /// `reinterpret_cast`): e.g. the storage of a variant's alternative.
    pub const fn cast<U>(&self) -> GuestPtr<U> {
        GuestPtr::new(self.addr)
    }
}

impl<T> Clone for GuestPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GuestPtr<T> {}
impl<T> PartialEq for GuestPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
    }
}
impl<T> Eq for GuestPtr<T> {}
impl<T> std::fmt::Debug for GuestPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GuestPtr({:#x})", self.addr)
    }
}

impl<T: GuestPod> GuestPtr<T> {
    fn check_alignment(&self) -> Result<()> {
        if T::ALIGN > 1 && !self.addr.is_multiple_of(T::ALIGN as u64) {
            return Err(MachineError::trigger(
                ExceptionKind::InvalidAlignment,
                self.addr,
            ));
        }
        Ok(())
    }
    /// Copy the object out of guest memory.
    pub fn get(&self, m: &Machine) -> Result<T> {
        self.check_alignment()?;
        m.memory.read_value(self.addr)
    }
    /// Overwrite the object in guest memory (bytewise, like a C++
    /// assignment of the mirror: no freeing, no relocation).
    pub fn set(&self, m: &mut Machine, value: &T) -> Result<()> {
        self.check_alignment()?;
        m.memory.write_value(self.addr, value)
    }
}

impl<T: GuestType> GuestPtr<T> {
    /// Construct the object from a host value (C++
    /// `construct_guest_object`). Whatever was there is overwritten, not
    /// freed.
    pub fn init<A: GuestInit<T> + ?Sized>(&self, m: &mut Machine, value: &A) -> Result<()> {
        self.check_alignment()?;
        value.init_at(m, self.addr)
    }
    /// Replace the object with a host value, releasing what it owned (the
    /// C++ `ScopedArenaObject` assignments).
    pub fn assign<A: GuestInit<T> + ?Sized>(&self, m: &mut Machine, value: &A) -> Result<()> {
        self.check_alignment()?;
        value.assign_at(m, self.addr)
    }
    /// Release the guest memory the object owns (C++ `free_guest_object`).
    pub fn free(&self, m: &mut Machine) -> Result<()> {
        T::free_at(m, self.addr)
    }
    /// Tell the object that it now lives here, after a bytewise move (C++
    /// `relocate_guest_object`).
    pub fn relocate(&self, m: &mut Machine) -> Result<()> {
        T::relocate_at(m, self.addr)
    }
}

/// A guest object type: a [`GuestPod`] with the container operations that
/// C++ selects with `is_guest_datatype` and
/// `is_self_referencing_guest_object`. Plain values use the defaults (they
/// own nothing and don't point into themselves).
pub trait GuestType: GuestPod {
    /// C++ `is_self_referencing_guest_object`: the object points back into
    /// itself and must be told when it moves.
    const SELF_REFERENCING: bool = false;
    /// Release the guest memory owned by the object at `addr` (C++
    /// `free_guest_object`). The object stays valid and empty.
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        let _ = (m, addr);
        Ok(())
    }
    /// The object at `addr` was copied there bytewise: update the pointers
    /// into itself (C++ `relocate_guest_object`).
    fn relocate_at(m: &mut Machine, addr: u64) -> Result<()> {
        let _ = (m, addr);
        Ok(())
    }
    /// Release the guest memory owned by this host-side value (C++
    /// `free(machine)` on an object that isn't in guest memory, e.g. a
    /// variant built on the host and passed by value). The value stays
    /// valid and empty.
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        let _ = m;
        Ok(())
    }
}

macro_rules! plain_types {
    ($($t:ty),*) => {$( impl GuestType for $t {} )*};
}
plain_types!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
impl<T: GuestPod, const N: usize> GuestType for [T; N] {}

/// C++ `bool`: one byte.
impl GuestPod for bool {
    const SIZE: usize = 1;
    const ALIGN: usize = 1;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
    fn read_guest(bytes: &[u8]) -> Self {
        bytes.first().is_some_and(|&b| b != 0)
    }
}

/// A guest value that can be copied to the host on its own (C++
/// `guest_host_type` and `guest_object_to_host`): plain values as they
/// are, a string as a `String`. Nested containers are read element by
/// element instead.
pub trait ToHost {
    type Host;
    fn to_host(&self, m: &Machine) -> Result<Self::Host>;
}

macro_rules! plain_to_host {
    ($($t:ty),*) => {$(
        impl ToHost for $t {
            type Host = $t;
            fn to_host(&self, _m: &Machine) -> Result<$t> {
                Ok(*self)
            }
        }
    )*};
}
plain_to_host!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);

impl ToHost for GuestStdString {
    type Host = String;
    fn to_host(&self, m: &Machine) -> Result<String> {
        self.to_string(m)
    }
}

/// A host value a `T` can be constructed from at a guest address (C++
/// `construct_guest_object`, and the constructors of the containers that
/// take `(machine, self, value)`). Every [`GuestType`] constructs from
/// itself: a shallow copy, relocated, that takes over the guest memory
/// the original owned.
pub trait GuestInit<T: GuestType> {
    /// Construct a `T` at `addr`, which holds no live object.
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()>;
    /// Replace the live `T` at `addr` (default: free it, then construct).
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        T::free_at(m, addr)?;
        self.init_at(m, addr)
    }
}

impl<T: GuestType> GuestInit<T> for T {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        m.memory.write_value(addr, self)?;
        T::relocate_at(m, addr)
    }
}

/// Owns a `T` on the native heap (the machine's arena), together with
/// every guest allocation it holds (C++ `ScopedArenaObject`). Dereferences
/// to its [`GuestPtr`], whose methods access and change the object.
///
/// C++ frees the object in its destructor. Here, call [`free`](Self::free)
/// when done; dropping it without leaks the guest memory (not host memory).
#[must_use = "a ScopedArenaObject leaks guest memory unless free() is called"]
#[derive(Debug)]
pub struct ScopedArenaObject<T: GuestType> {
    ptr: GuestPtr<T>,
}

impl<T: GuestType> ScopedArenaObject<T> {
    /// Allocate a `T` on the native heap and construct it from `value`.
    pub fn with<A: GuestInit<T> + ?Sized>(m: &mut Machine, value: &A) -> Result<Self> {
        let addr = guest_alloc(m, T::SIZE)?;
        if let Err(e) = value.init_at(m, addr) {
            let _ = guest_free(m, addr);
            return Err(e);
        }
        Ok(Self {
            ptr: GuestPtr::new(addr),
        })
    }

    /// Allocate a default-constructed `T` on the native heap.
    pub fn new(m: &mut Machine) -> Result<Self>
    where
        T: Default,
    {
        Self::with(m, &T::default())
    }

    /// The object's guest address.
    pub const fn address(&self) -> u64 {
        self.ptr.addr
    }

    /// The handle of the object.
    pub const fn ptr(&self) -> GuestPtr<T> {
        self.ptr
    }

    /// Release the object's guest memory and the object itself.
    pub fn free(self, m: &mut Machine) -> Result<()> {
        T::free_at(m, self.ptr.addr)?;
        guest_free(m, self.ptr.addr)
    }
}

impl<T: GuestType> std::ops::Deref for ScopedArenaObject<T> {
    type Target = GuestPtr<T>;
    fn deref(&self) -> &GuestPtr<T> {
        &self.ptr
    }
}

/// A scoped object is passed by its address (C++ `is_scoped_guest_object`
/// in `setup_call`).
impl<T: GuestType> VmCallArg for ScopedArenaObject<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.int(m, self.ptr.addr)
    }
}

/// A handle is passed as the address of its object.
impl<T> VmCallArg for GuestPtr<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.int(m, self.addr)
    }
}

/// A handle is read from a system call argument holding its address (C++
/// `sysargs<T*>` followed by `memarray`).
impl<T> crate::sysargs::SysArg<'_> for GuestPtr<T> {
    fn read(m: &Machine, args: &mut crate::sysargs::ArgCursor) -> Result<Self> {
        Ok(Self::new(args.next_int(m)))
    }
}

/// C++ `setup_call` for a guest container passed by value: it is pushed
/// onto the stack, and told where it ended up there (the small-string
/// optimization and the map buckets point into the object).
pub(crate) fn pass_by_value<T: GuestType>(
    value: &T,
    m: &mut Machine,
    regs: &mut ArgRegs,
) -> Result<()> {
    let addr = m.stack_push_value(value)?;
    T::relocate_at(m, addr)?;
    regs.int(m, addr)
}
