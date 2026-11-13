//! Port of `guest/guest_cpp_unordered_map.hpp`: libstdc++'s
//! `std::unordered_map` (`_Hashtable` with unique keys).

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use super::hash::{FAST_BKT, PRIMES};
use super::{
    GuestHashKey, GuestInit, GuestKeyArg, GuestPtr, GuestType, HashPolicy, StdHash, ToHost,
    container_error, guest_alloc, guest_free, pass_by_value, read_u64, round_up, write_u64,
};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::vmcall::{ArgRegs, VmCallArg};

/// libstdc++'s `std::unordered_map<K, V>` (C++ `GuestStdUnorderedMap`), 7
/// guest words: the bucket array, the bucket count, `_M_before_begin` (the
/// head of the list of every node), the element count, the rehash policy
/// (max load factor, next resize) and `_M_single_bucket`.
///
/// A bucket doesn't point at its first node but at the node *before* it,
/// so the bucket of the first node points at `before_begin`, inside the
/// map; a map with one bucket uses `single_bucket` as its bucket array.
/// So a map must be told when it moves ([`GuestPtr::relocate`]), and
/// every operation works in place, through a [`GuestPtr`] at the map's
/// address. A host-side value (`default()`) is only usable by the guest
/// once placed somewhere.
///
/// `H` says whether the nodes store hash codes, which depends on the hash
/// function: [`StdHash`] for `std::hash<K>`, [`CustomHash`](super::CustomHash)
/// for any other (which must still compute the same values). Getting it
/// wrong shifts every key and value in the node.
pub struct GuestStdUnorderedMap<K, V, H = StdHash> {
    pub buckets: u64,
    pub bucket_cnt: u64,
    pub before_begin: u64,
    pub element_cnt: u64,
    pub max_load: f32,
    pub next_resize: u64,
    pub single_bucket: u64,
    _types: PhantomData<MapTypes<K, V, H>>,
}

type MapTypes<K, V, H> = fn() -> (K, V, H);

const BUCKETS: u64 = 0;
const BUCKET_CNT: u64 = 8;
const BEFORE_BEGIN_OFF: u64 = 16;
const ELEMENT_CNT: u64 = 24;
const MAX_LOAD: u64 = 32;
const NEXT_RESIZE: u64 = 40;
const SINGLE_BUCKET_OFF: u64 = 48;

impl<K, V, H> Default for GuestStdUnorderedMap<K, V, H> {
    /// An empty map (C++ `GuestStdUnorderedMap()`): one bucket, which
    /// becomes `single_bucket` when it gets an address.
    fn default() -> Self {
        Self {
            buckets: 0,
            bucket_cnt: 1,
            before_begin: 0,
            element_cnt: 0,
            max_load: 1.0,
            next_resize: 0,
            single_bucket: 0,
            _types: PhantomData,
        }
    }
}
impl<K, V, H> Clone for GuestStdUnorderedMap<K, V, H> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, V, H> Copy for GuestStdUnorderedMap<K, V, H> {}
impl<K, V, H> PartialEq for GuestStdUnorderedMap<K, V, H> {
    fn eq(&self, o: &Self) -> bool {
        self.buckets == o.buckets
            && self.bucket_cnt == o.bucket_cnt
            && self.before_begin == o.before_begin
            && self.element_cnt == o.element_cnt
            && self.max_load.to_bits() == o.max_load.to_bits()
            && self.next_resize == o.next_resize
            && self.single_bucket == o.single_bucket
    }
}
impl<K, V, H> std::fmt::Debug for GuestStdUnorderedMap<K, V, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestStdUnorderedMap")
            .field("buckets", &self.buckets)
            .field("bucket_cnt", &self.bucket_cnt)
            .field("before_begin", &self.before_begin)
            .field("element_cnt", &self.element_cnt)
            .field("max_load", &self.max_load)
            .field("next_resize", &self.next_resize)
            .field("single_bucket", &self.single_bucket)
            .finish()
    }
}

impl<K, V, H> GuestPod for GuestStdUnorderedMap<K, V, H> {
    const SIZE: usize = 56;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        for w in [
            self.buckets,
            self.bucket_cnt,
            self.before_begin,
            self.element_cnt,
        ] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&self.max_load.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&self.next_resize.to_le_bytes());
        out.extend_from_slice(&self.single_bucket.to_le_bytes());
    }
    fn read_guest(bytes: &[u8]) -> Self {
        let w = |off: usize| u64::read_guest(bytes.get(off..).unwrap_or_default());
        Self {
            buckets: w(0),
            bucket_cnt: w(8),
            before_begin: w(16),
            element_cnt: w(24),
            max_load: f32::read_guest(bytes.get(32..).unwrap_or_default()),
            next_resize: w(40),
            single_bucket: w(48),
            _types: PhantomData,
        }
    }
}

/// std::floor for non-negative values.
fn floor_size(value: f64) -> u64 {
    if value > 0.0 { value as u64 } else { 0 }
}
/// std::ceil for non-negative values. An infinite value (a guest's zero
/// load factor) wraps around to 0, like C++'s `size_t` arithmetic.
fn ceil_size(value: f64) -> u64 {
    let result = floor_size(value);
    if (result as f64) < value {
        result.wrapping_add(1)
    } else {
        result
    }
}

fn corrupt_list() -> crate::MachineError {
    container_error("Guest std::unordered_map: corrupt node list")
}

/// Cycle detection for a walk over guest nodes (Brent's algorithm). The
/// guest controls the `_M_nxt` pointers, and bounding a walk by the
/// guest's element count isn't enough: a cyclic list with a huge count
/// kept the host busy (and collecting entries) all but forever. Not in
/// C++, which loops there. Constant memory; a list without a cycle never
/// fails the check.
struct CycleCheck {
    tortoise: u64,
    power: u64,
    steps: u64,
}

impl CycleCheck {
    const fn new(start: u64) -> Self {
        Self {
            tortoise: start,
            power: 1,
            steps: 0,
        }
    }
    /// Called with each node the walk moves to: an error once it has come
    /// back to a node it visited.
    fn step(&mut self, addr: u64) -> Result<()> {
        if addr != 0 && addr == self.tortoise {
            return Err(corrupt_list());
        }
        self.steps += 1;
        if self.steps == self.power {
            self.tortoise = addr;
            self.power = self.power.saturating_mul(2);
            self.steps = 0;
        }
        Ok(())
    }
}

/// The number of nodes in the list starting at `first`, which must be
/// acyclic and no longer than `max` (the element count). The walks that
/// change nodes as they go check the list with this first: their own
/// writes could change where a cycle leads.
fn node_list_len(m: &Machine, first: u64, max: u64) -> Result<u64> {
    let mut check = CycleCheck::new(first);
    let mut addr = first;
    let mut len = 0u64;
    while addr != 0 {
        len += 1;
        if len > max {
            return Err(corrupt_list());
        }
        addr = read_u64(m, addr)?;
        check.step(addr)?;
    }
    Ok(len)
}

impl<K: GuestHashKey, V: GuestType, H: HashPolicy<K>> GuestStdUnorderedMap<K, V, H> {
    /// True when the guest stores the hash code of each key in its node.
    pub const CACHE_HASH_CODE: bool = H::CACHE_HASH_CODE;
    /// Offset of the key in a node (`_Hash_node`: the next pointer, then
    /// the `std::pair<const K, V>`, then the cached hash code).
    pub const KEY_OFF: u64 = round_up(8, K::ALIGN) as u64;
    const PAIR_ALIGN: usize = if K::ALIGN > V::ALIGN {
        K::ALIGN
    } else {
        V::ALIGN
    };
    const PAIR_SIZE: usize = round_up(round_up(K::SIZE, V::ALIGN) + V::SIZE, Self::PAIR_ALIGN);
    /// Offset of the value in a node.
    pub const VALUE_OFF: u64 = Self::KEY_OFF + round_up(K::SIZE, V::ALIGN) as u64;
    const HASH_OFF: u64 = round_up(Self::KEY_OFF as usize + Self::PAIR_SIZE, 8) as u64;
    const NODE_ALIGN: usize = if Self::PAIR_ALIGN > 8 {
        Self::PAIR_ALIGN
    } else {
        8
    };
    /// `sizeof(node_type)`.
    pub const NODE_SIZE: usize = round_up(
        if Self::CACHE_HASH_CODE {
            Self::HASH_OFF as usize + 8
        } else {
            Self::KEY_OFF as usize + Self::PAIR_SIZE
        },
        Self::NODE_ALIGN,
    );

    pub const fn size(&self) -> usize {
        self.element_cnt as usize
    }
    pub const fn empty(&self) -> bool {
        self.element_cnt == 0
    }
    pub const fn bucket_count(&self) -> usize {
        self.bucket_cnt as usize
    }
    pub const fn max_load_factor(&self) -> f32 {
        self.max_load
    }
    pub fn load_factor(&self) -> f32 {
        self.element_cnt as f32 / self.bucket_cnt as f32
    }
    /// `bucket_cnt * sizeof(word) + element_cnt * sizeof(node)`.
    pub const fn size_bytes(&self) -> u64 {
        self.bucket_cnt
            .wrapping_mul(8)
            .wrapping_add(self.element_cnt.wrapping_mul(Self::NODE_SIZE as u64))
    }
}

/// Construct a map from a host map (C++ `GuestStdUnorderedMap(machine,
/// self, container)`). Assigning one keeps the bucket array, like C++
/// `assign`.
impl<K, V, H, HK, HV, S> GuestInit<GuestStdUnorderedMap<K, V, H>> for HashMap<HK, HV, S>
where
    K: GuestHashKey,
    V: GuestType,
    H: HashPolicy<K>,
    HK: GuestKeyArg<K> + GuestInit<K>,
    HV: GuestInit<V>,
    S: BuildHasher,
{
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<GuestStdUnorderedMap<K, V, H>>::new(addr);
        GuestStdUnorderedMap::<K, V, H>::default().init_at(m, addr)?;
        ptr.assign_entries(m, self)
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<GuestStdUnorderedMap<K, V, H>>::new(addr).assign_entries(m, self)
    }
}

/// Construct a map from key/value pairs, inserted in order.
impl<K, V, H, HK, HV> GuestInit<GuestStdUnorderedMap<K, V, H>> for [(HK, HV)]
where
    K: GuestHashKey,
    V: GuestType,
    H: HashPolicy<K>,
    HK: GuestKeyArg<K> + GuestInit<K>,
    HV: GuestInit<V>,
{
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<GuestStdUnorderedMap<K, V, H>>::new(addr);
        GuestStdUnorderedMap::<K, V, H>::default().init_at(m, addr)?;
        ptr.assign_entries(m, self.iter().map(|(k, v)| (k, v)))
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<GuestStdUnorderedMap<K, V, H>>::new(addr);
        ptr.assign_entries(m, self.iter().map(|(k, v)| (k, v)))
    }
}

impl<K: GuestHashKey, V: GuestType, H: HashPolicy<K>> GuestType for GuestStdUnorderedMap<K, V, H> {
    const SELF_REFERENCING: bool = true;
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).free_all(m)
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        let mut addr = self.before_begin;
        let limit = node_list_len(m, addr, self.element_cnt)?;
        let mut freed = 0u64;
        while addr != 0 {
            let next = read_u64(m, addr)?;
            K::free_at(m, addr.wrapping_add(Self::KEY_OFF))?;
            V::free_at(m, addr.wrapping_add(Self::VALUE_OFF))?;
            guest_free(m, addr)?;
            addr = next;
            freed += 1;
            if freed > limit {
                return Err(corrupt_list());
            }
        }
        if self.bucket_cnt != 1 && self.buckets != 0 {
            guest_free(m, self.buckets)?;
        }
        *self = Self::default();
        Ok(())
    }
    fn relocate_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).move_to_self(m)
    }
}

type Map<K, V, H> = GuestStdUnorderedMap<K, V, H>;

impl<K: GuestHashKey, V: GuestType, H: HashPolicy<K>> GuestPtr<Map<K, V, H>> {
    // --- the map's own fields, in guest memory ---------------------------

    fn field(&self, m: &Machine, off: u64) -> Result<u64> {
        read_u64(m, self.address().wrapping_add(off))
    }
    fn set_field(&self, m: &mut Machine, off: u64, v: u64) -> Result<()> {
        write_u64(m, self.address().wrapping_add(off), v)
    }
    fn before_begin(&self, m: &Machine) -> Result<u64> {
        self.field(m, BEFORE_BEGIN_OFF)
    }
    fn element_cnt(&self, m: &Machine) -> Result<u64> {
        self.field(m, ELEMENT_CNT)
    }
    fn bucket_cnt(&self, m: &Machine) -> Result<u64> {
        self.field(m, BUCKET_CNT)
    }
    fn max_load(&self, m: &Machine) -> Result<f32> {
        Ok(f32::from_bits(
            m.memory.read_u32(self.address().wrapping_add(MAX_LOAD))?,
        ))
    }

    pub fn size(&self, m: &Machine) -> Result<usize> {
        Ok(self.element_cnt(m)? as usize)
    }
    pub fn empty(&self, m: &Machine) -> Result<bool> {
        Ok(self.element_cnt(m)? == 0)
    }
    pub fn bucket_count(&self, m: &Machine) -> Result<usize> {
        Ok(self.bucket_cnt(m)? as usize)
    }
    pub fn max_load_factor(&self, m: &Machine) -> Result<f32> {
        self.max_load(m)
    }
    pub fn load_factor(&self, m: &Machine) -> Result<f32> {
        Ok(self.get(m)?.load_factor())
    }
    /// C++ `set_max_load_factor`.
    pub fn set_max_load_factor(&self, m: &mut Machine, mlf: f32) -> Result<()> {
        m.memory
            .write_u32(self.address().wrapping_add(MAX_LOAD), mlf.to_bits())?;
        let cnt = self.bucket_cnt(m)?;
        if cnt > 1 {
            self.set_field(m, NEXT_RESIZE, floor_size(cnt as f64 * f64::from(mlf)))?;
        }
        Ok(())
    }

    // --- nodes and buckets ------------------------------------------------

    /// The bucket array: `single_bucket` when there is one bucket. Fails
    /// for a map that was never given an address.
    fn bucket_array(&self, m: &Machine) -> Result<u64> {
        let buckets = self.field(m, BUCKETS)?;
        if buckets == 0 {
            return Err(container_error(
                "Guest std::unordered_map has no address in guest memory (see move())",
            ));
        }
        Ok(buckets)
    }
    fn bucket(&self, m: &Machine, bkts: u64, bkt: u64) -> Result<u64> {
        read_u64(m, bkts.wrapping_add(bkt.wrapping_mul(8)))
    }
    fn set_bucket(&self, m: &mut Machine, bkts: u64, bkt: u64, v: u64) -> Result<()> {
        write_u64(m, bkts.wrapping_add(bkt.wrapping_mul(8)), v)
    }
    /// The bucket a hash code belongs to (`_Mod_range_hashing`).
    fn bucket_index(&self, m: &Machine, code: u64) -> Result<u64> {
        let cnt = self.bucket_cnt(m)?;
        if cnt == 0 {
            return Err(container_error("Guest std::unordered_map has zero buckets"));
        }
        Ok(code % cnt)
    }
    /// The `_M_nxt` of a node, or of `before_begin`.
    fn next(&self, m: &Machine, node_base: u64) -> Result<u64> {
        read_u64(m, node_base)
    }
    fn set_next(&self, m: &mut Machine, node_base: u64, v: u64) -> Result<()> {
        write_u64(m, node_base, v)
    }
    fn key_at(&self, m: &Machine, node: u64) -> Result<K> {
        m.memory
            .read_value(node.wrapping_add(Map::<K, V, H>::KEY_OFF))
    }
    /// The hash code of a node's key, cached or recomputed.
    fn node_hash(&self, m: &Machine, node: u64) -> Result<u64> {
        if Map::<K, V, H>::CACHE_HASH_CODE {
            read_u64(m, node.wrapping_add(Map::<K, V, H>::HASH_OFF))
        } else {
            self.key_at(m, node)?.guest_hash(m)
        }
    }
    fn node_equals<A: GuestKeyArg<K> + ?Sized>(
        &self,
        m: &Machine,
        node: u64,
        key: &A,
        code: u64,
    ) -> Result<bool> {
        if Map::<K, V, H>::CACHE_HASH_CODE
            && read_u64(m, node.wrapping_add(Map::<K, V, H>::HASH_OFF))? != code
        {
            return Ok(false);
        }
        key.key_eq(m, &self.key_at(m, node)?)
    }

    /// The node *before* the node of `key`, as stored in the buckets
    /// (`_M_find_before_node`), or 0.
    fn find_before_node<A: GuestKeyArg<K> + ?Sized>(
        &self,
        m: &Machine,
        bkt: u64,
        key: &A,
        code: u64,
    ) -> Result<u64> {
        let bkts = self.bucket_array(m)?;
        let mut prev = self.bucket(m, bkts, bkt)?;
        if prev == 0 {
            return Ok(0);
        }
        let mut addr = self.next(m, prev)?;
        let mut check = CycleCheck::new(addr);
        while addr != 0 {
            if self.node_equals(m, addr, key, code)? {
                return Ok(prev);
            }
            // Stop at the end of the list, or when leaving this bucket
            let next = self.next(m, addr)?;
            if next == 0 || self.bucket_index(m, self.node_hash(m, next)?)? != bkt {
                break;
            }
            prev = addr;
            addr = next;
            check.step(addr)?;
        }
        Ok(0)
    }

    /// The guest address of the node of `key`, or 0.
    pub fn find_node<A: GuestKeyArg<K> + ?Sized>(&self, m: &Machine, key: &A) -> Result<u64> {
        if self.element_cnt(m)? == 0 {
            return Ok(0);
        }
        let code = key.key_hash(m)?;
        let prev = self.find_before_node(m, self.bucket_index(m, code)?, key, code)?;
        if prev != 0 { self.next(m, prev) } else { Ok(0) }
    }

    // --- lookups ------------------------------------------------------------

    /// The value of `key`, or `None` (C++ `find`).
    pub fn find<A: GuestKeyArg<K> + ?Sized>(
        &self,
        m: &Machine,
        key: &A,
    ) -> Result<Option<GuestPtr<V>>> {
        let node = self.find_node(m, key)?;
        Ok((node != 0).then(|| GuestPtr::new(node.wrapping_add(Map::<K, V, H>::VALUE_OFF))))
    }
    /// The value of `key`, or an error (C++ `at`).
    pub fn at<A: GuestKeyArg<K> + ?Sized>(&self, m: &Machine, key: &A) -> Result<GuestPtr<V>> {
        self.find(m, key)?
            .ok_or(container_error("Guest std::unordered_map: Key not found"))
    }
    pub fn contains<A: GuestKeyArg<K> + ?Sized>(&self, m: &Machine, key: &A) -> Result<bool> {
        Ok(self.find_node(m, key)? != 0)
    }
    pub fn count<A: GuestKeyArg<K> + ?Sized>(&self, m: &Machine, key: &A) -> Result<usize> {
        Ok(usize::from(self.contains(m, key)?))
    }

    /// Every key (a copy) and value, in list order. Collected first, so
    /// the caller may change the map's values.
    pub fn entries(&self, m: &Machine) -> Result<Vec<(K, GuestPtr<V>)>> {
        let mut out = Vec::with_capacity(self.element_cnt(m)?.min(1 << 20) as usize);
        let mut addr = self.before_begin(m)?;
        let mut check = CycleCheck::new(addr);
        while addr != 0 {
            out.push((
                self.key_at(m, addr)?,
                GuestPtr::new(addr.wrapping_add(Map::<K, V, H>::VALUE_OFF)),
            ));
            addr = self.next(m, addr)?;
            // A corrupt (cyclic) list must still end.
            check.step(addr)?;
            if out.len() as u64 > self.element_cnt(m)? {
                return Err(corrupt_list());
            }
        }
        Ok(out)
    }

    /// Visit every element, in an unspecified order (C++ `for_each`).
    pub fn for_each<F: FnMut(&K, GuestPtr<V>) -> Result<()>>(
        &self,
        m: &Machine,
        mut callback: F,
    ) -> Result<()> {
        for (k, v) in self.entries(m)? {
            callback(&k, v)?;
        }
        Ok(())
    }

    /// Copy the whole map to the host (C++ `to_map`).
    pub fn to_map(&self, m: &Machine) -> Result<HashMap<K::Host, V::Host>>
    where
        K: ToHost,
        V: ToHost,
        K::Host: Eq + Hash,
    {
        let mut result = HashMap::new();
        for (k, v) in self.entries(m)? {
            result.insert(k.to_host(m)?, v.get(m)?.to_host(m)?);
        }
        Ok(result)
    }

    // --- changes ------------------------------------------------------------

    /// Insert a key/value pair, overwriting the value of an existing key
    /// (C++ `insert_or_assign`). Returns the node's guest address.
    pub fn insert_or_assign<A, B>(&self, m: &mut Machine, key: &A, value: &B) -> Result<u64>
    where
        A: GuestKeyArg<K> + GuestInit<K> + ?Sized,
        B: GuestInit<V> + ?Sized,
    {
        let code = key.key_hash(m)?;
        if self.element_cnt(m)? != 0 {
            let bkt = self.bucket_index(m, code)?;
            let prev = self.find_before_node(m, bkt, key, code)?;
            if prev != 0 {
                let addr = self.next(m, prev)?;
                let value_addr = addr.wrapping_add(Map::<K, V, H>::VALUE_OFF);
                V::free_at(m, value_addr)?;
                value.init_at(m, value_addr)?;
                return Ok(addr);
            }
        }
        // Grow the bucket array first, exactly like libstdc++ does
        self.maybe_rehash(m, 1)?;
        let addr = guest_alloc(m, Map::<K, V, H>::NODE_SIZE)?;
        self.set_next(m, addr, 0)?;
        if Map::<K, V, H>::CACHE_HASH_CODE {
            write_u64(m, addr.wrapping_add(Map::<K, V, H>::HASH_OFF), code)?;
        }
        key.init_at(m, addr.wrapping_add(Map::<K, V, H>::KEY_OFF))?;
        value.init_at(m, addr.wrapping_add(Map::<K, V, H>::VALUE_OFF))?;
        let bkt = self.bucket_index(m, code)?;
        self.insert_bucket_begin(m, bkt, addr)?;
        let cnt = self.element_cnt(m)?;
        self.set_field(m, ELEMENT_CNT, cnt.wrapping_add(1))?;
        Ok(addr)
    }

    /// Insert a key/value pair unless the key exists (C++ `try_emplace`).
    /// Returns the node's guest address.
    pub fn try_emplace<A, B>(&self, m: &mut Machine, key: &A, value: &B) -> Result<u64>
    where
        A: GuestKeyArg<K> + GuestInit<K> + ?Sized,
        B: GuestInit<V> + ?Sized,
    {
        let addr = self.find_node(m, key)?;
        if addr != 0 {
            return Ok(addr);
        }
        self.insert_or_assign(m, key, value)
    }

    /// Remove `key`; true if there was one (C++ `erase`).
    pub fn erase<A: GuestKeyArg<K> + ?Sized>(&self, m: &mut Machine, key: &A) -> Result<bool> {
        if self.element_cnt(m)? == 0 {
            return Ok(false);
        }
        let code = key.key_hash(m)?;
        let bkt = self.bucket_index(m, code)?;
        let prev = self.find_before_node(m, bkt, key, code)?;
        if prev == 0 {
            return Ok(false);
        }
        let addr = self.next(m, prev)?;
        let next = self.next(m, addr)?;
        // The bucket of the next node decides whether this bucket empties
        let next_bkt = if next != 0 {
            self.bucket_index(m, self.node_hash(m, next)?)?
        } else {
            bkt
        };
        K::free_at(m, addr.wrapping_add(Map::<K, V, H>::KEY_OFF))?;
        V::free_at(m, addr.wrapping_add(Map::<K, V, H>::VALUE_OFF))?;
        guest_free(m, addr)?;
        let cnt = self.element_cnt(m)?;
        self.set_field(m, ELEMENT_CNT, cnt.wrapping_sub(1))?;
        let bkts = self.bucket_array(m)?;
        if prev == self.bucket(m, bkts, bkt)? {
            self.remove_bucket_begin(m, bkt, next, next_bkt)?;
        } else if next_bkt != bkt {
            self.set_bucket(m, bkts, next_bkt, prev)?;
        }
        self.set_next(m, prev, next)?;
        Ok(true)
    }

    /// Remove every element, keeping the bucket array (C++ `clear`).
    pub fn clear(&self, m: &mut Machine) -> Result<()> {
        let mut addr = self.before_begin(m)?;
        let mut freed = 0u64;
        let limit = node_list_len(m, addr, self.element_cnt(m)?)?;
        while addr != 0 {
            let next = self.next(m, addr)?;
            K::free_at(m, addr.wrapping_add(Map::<K, V, H>::KEY_OFF))?;
            V::free_at(m, addr.wrapping_add(Map::<K, V, H>::VALUE_OFF))?;
            guest_free(m, addr)?;
            addr = next;
            freed += 1;
            if freed > limit {
                return Err(corrupt_list());
            }
        }
        self.set_field(m, BEFORE_BEGIN_OFF, 0)?;
        self.set_field(m, ELEMENT_CNT, 0)?;
        let buckets = self.field(m, BUCKETS)?;
        if buckets != 0 {
            let len = self.bucket_cnt(m)?.saturating_mul(8);
            m.memory.memset(buckets, 0, len)?;
        }
        Ok(())
    }

    /// Free every element and the bucket array (C++ `free`). The map is
    /// detached from guest memory afterwards: relocate it to use it again.
    fn free_all(&self, m: &mut Machine) -> Result<()> {
        self.clear(m)?;
        let buckets = self.field(m, BUCKETS)?;
        if self.bucket_cnt(m)? != 1 && buckets != 0 {
            guest_free(m, buckets)?;
        }
        self.set_field(m, BUCKETS, 0)?;
        self.set_field(m, BUCKET_CNT, 1)?;
        self.set_field(m, SINGLE_BUCKET_OFF, 0)?;
        self.set_field(m, NEXT_RESIZE, 0)
    }

    /// The map now lives here (C++ `move`): update the pointers into it.
    fn move_to_self(&self, m: &mut Machine) -> Result<()> {
        let me = self.address();
        if self.bucket_cnt(m)? == 1 {
            // libstdc++ uses single_bucket as a 1-element bucket array
            self.set_field(m, BUCKETS, me.wrapping_add(SINGLE_BUCKET_OFF))?;
        }
        let first = self.before_begin(m)?;
        if first != 0 {
            // Exactly one bucket points at before_begin: the bucket of the
            // first node of the list.
            let bkts = self.bucket_array(m)?;
            let bkt = self.bucket_index(m, self.node_hash(m, first)?)?;
            self.set_bucket(m, bkts, bkt, me.wrapping_add(BEFORE_BEGIN_OFF))?;
        }
        Ok(())
    }

    /// Replace the contents with key/value pairs (C++ `assign`): clear,
    /// reserve, insert. (Assigning a `HashMap` with [`GuestPtr::assign`]
    /// does the same.)
    pub fn assign_entries<'a, I, A, B>(&self, m: &mut Machine, entries: I) -> Result<()>
    where
        I: IntoIterator<Item = (&'a A, &'a B)>,
        I::IntoIter: ExactSizeIterator,
        A: GuestKeyArg<K> + GuestInit<K> + ?Sized + 'a,
        B: GuestInit<V> + ?Sized + 'a,
    {
        let entries = entries.into_iter();
        self.clear(m)?;
        self.reserve(m, entries.len() as u64)?;
        for (k, v) in entries {
            self.insert_or_assign(m, k, v)?;
        }
        Ok(())
    }

    /// Set the bucket count to at least `count`, rehashing (C++ `rehash`,
    /// mirroring `std::unordered_map::rehash`).
    pub fn rehash(&self, m: &mut Machine, count: u64) -> Result<()> {
        let saved_resize = self.field(m, NEXT_RESIZE)?;
        let need = self.bkt_for_elements(m, self.element_cnt(m)?.wrapping_add(1))?;
        let buckets = self.next_bucket_count(m, need.max(count))?;
        if buckets != self.bucket_cnt(m)? {
            self.rehash_buckets(m, buckets)
        } else {
            // No rehash: keep the resize threshold consistent with the state
            self.set_field(m, NEXT_RESIZE, saved_resize)
        }
    }

    /// Make room for `count` elements (C++ `reserve`).
    pub fn reserve(&self, m: &mut Machine, count: u64) -> Result<()> {
        let buckets = self.bkt_for_elements(m, count)?;
        self.rehash(m, buckets)
    }

    /// The bucket count needed for `count` elements.
    fn bkt_for_elements(&self, m: &Machine, count: u64) -> Result<u64> {
        Ok(ceil_size(count as f64 / f64::from(self.max_load(m)?)))
    }

    /// `_Prime_rehash_policy::_M_need_rehash`: grow the bucket array when
    /// inserting `n_ins` elements would exceed the load factor.
    fn maybe_rehash(&self, m: &mut Machine, n_ins: u64) -> Result<bool> {
        let cnt = self.element_cnt(m)?;
        let next_resize = self.field(m, NEXT_RESIZE)?;
        if cnt.wrapping_add(n_ins) <= next_resize {
            return Ok(false);
        }
        // A zero resize threshold means that no buckets have been
        // allocated yet: libstdc++ starts out with at least 11 buckets.
        let elements = cnt
            .wrapping_add(n_ins)
            .max(if next_resize == 0 { 11 } else { 0 });
        let max_load = f64::from(self.max_load(m)?);
        let min_buckets = elements as f64 / max_load;
        let bucket_cnt = self.bucket_cnt(m)?;
        if min_buckets >= bucket_cnt as f64 {
            // A guest's zero load factor makes this infinite, then 0.
            let want = floor_size(min_buckets)
                .wrapping_add(1)
                .max(bucket_cnt.wrapping_mul(2));
            let count = self.next_bucket_count(m, want)?;
            self.rehash_buckets(m, count)?;
            return Ok(true);
        }
        self.set_field(m, NEXT_RESIZE, floor_size(bucket_cnt as f64 * max_load))?;
        Ok(false)
    }

    /// The bucket count libstdc++ picks for `count` buckets, updating the
    /// resize threshold (`_M_next_bkt`).
    fn next_bucket_count(&self, m: &mut Machine, count: u64) -> Result<u64> {
        let max_load = f64::from(self.max_load(m)?);
        if let Some(&fast) = usize::try_from(count).ok().and_then(|c| FAST_BKT.get(c)) {
            if count == 0 {
                // Keep the threshold at zero, so the next insertion allocates
                return Ok(1);
            }
            self.set_field(m, NEXT_RESIZE, floor_size(f64::from(fast) * max_load))?;
            return Ok(u64::from(fast));
        }
        // The last prime is never selected by the search, as in libstdc++
        let last = PRIMES.len() - 1;
        let search = PRIMES.get(6..last).unwrap_or_default();
        let it = 6 + search.partition_point(|&p| u64::from(p) < count);
        let last_prime = u64::from(PRIMES.get(last).copied().unwrap_or_default());
        if it == last {
            // Never rehash again: the biggest bucket count is reached
            self.set_field(m, NEXT_RESIZE, u64::MAX)?;
            return Ok(count.max(last_prime));
        }
        let prime = u64::from(PRIMES.get(it).copied().unwrap_or_default());
        self.set_field(m, NEXT_RESIZE, floor_size(prime as f64 * max_load))?;
        Ok(prime)
    }

    /// Move every node into a new bucket array (`_M_rehash_aux` for unique
    /// keys).
    fn rehash_buckets(&self, m: &mut Machine, count: u64) -> Result<()> {
        // Before anything changes: relinking could turn a cycle into a
        // walk that never repeats.
        let limit = node_list_len(m, self.before_begin(m)?, self.element_cnt(m)?)?;
        let me = self.address();
        let bbegin = me.wrapping_add(BEFORE_BEGIN_OFF);
        let new_buckets = if count == 1 {
            self.set_field(m, SINGLE_BUCKET_OFF, 0)?;
            me.wrapping_add(SINGLE_BUCKET_OFF)
        } else {
            let bytes = count
                .checked_mul(8)
                .and_then(|b| usize::try_from(b).ok())
                .ok_or(container_error(
                    "Guest std::unordered_map: too many buckets",
                ))?;
            let b = guest_alloc(m, bytes)?;
            m.memory.memset(b, 0, bytes as u64)?;
            b
        };
        // Re-insert every node, which also reverses the order of the list
        let mut addr = self.before_begin(m)?;
        self.set_field(m, BEFORE_BEGIN_OFF, 0)?;
        let mut bbegin_bkt = 0;
        let mut moved = 0u64;
        while addr != 0 {
            let next = self.next(m, addr)?;
            let bkt = self.node_hash(m, addr)? % count;
            let head = self.bucket(m, new_buckets, bkt)?;
            if head == 0 {
                let first = self.before_begin(m)?;
                self.set_next(m, addr, first)?;
                self.set_field(m, BEFORE_BEGIN_OFF, addr)?;
                self.set_bucket(m, new_buckets, bkt, bbegin)?;
                if first != 0 {
                    self.set_bucket(m, new_buckets, bbegin_bkt, addr)?;
                }
                bbegin_bkt = bkt;
            } else {
                let prev_next = self.next(m, head)?;
                self.set_next(m, addr, prev_next)?;
                self.set_next(m, head, addr)?;
            }
            addr = next;
            moved += 1;
            if moved > limit {
                return Err(corrupt_list());
            }
        }
        let old = self.field(m, BUCKETS)?;
        if self.bucket_cnt(m)? != 1 && old != 0 {
            guest_free(m, old)?;
        }
        self.set_field(m, BUCKET_CNT, count)?;
        self.set_field(m, BUCKETS, new_buckets)
    }

    /// Insert a node at the beginning of a bucket (`_M_insert_bucket_begin`).
    fn insert_bucket_begin(&self, m: &mut Machine, bkt: u64, addr: u64) -> Result<()> {
        let bkts = self.bucket_array(m)?;
        let head = self.bucket(m, bkts, bkt)?;
        if head != 0 {
            // Not empty: insert after the node before the bucket
            let prev_next = self.next(m, head)?;
            self.set_next(m, addr, prev_next)?;
            self.set_next(m, head, addr)
        } else {
            // Empty: the node becomes the first of the list, and the bucket
            // points at before_begin
            let first = self.before_begin(m)?;
            self.set_next(m, addr, first)?;
            self.set_field(m, BEFORE_BEGIN_OFF, addr)?;
            if first != 0 {
                // Update the bucket that pointed at before_begin
                let old_bkt = self.bucket_index(m, self.node_hash(m, first)?)?;
                self.set_bucket(m, bkts, old_bkt, addr)?;
            }
            let bbegin = self.address().wrapping_add(BEFORE_BEGIN_OFF);
            self.set_bucket(m, bkts, bkt, bbegin)
        }
    }

    /// Unlink the first node of a bucket (`_M_remove_bucket_begin`).
    fn remove_bucket_begin(
        &self,
        m: &mut Machine,
        bkt: u64,
        next: u64,
        next_bkt: u64,
    ) -> Result<()> {
        if next == 0 || next_bkt != bkt {
            // The bucket is now empty
            let bkts = self.bucket_array(m)?;
            if next != 0 {
                let head = self.bucket(m, bkts, bkt)?;
                self.set_bucket(m, bkts, next_bkt, head)?;
            }
            if self.bucket(m, bkts, bkt)? == self.address().wrapping_add(BEFORE_BEGIN_OFF) {
                self.set_field(m, BEFORE_BEGIN_OFF, next)?;
            }
            self.set_bucket(m, bkts, bkt, 0)?;
        }
        Ok(())
    }
}

/// By value: pushed onto the stack, and relocated there (C++ `setup_call`).
impl<K: GuestHashKey, V: GuestType, H: HashPolicy<K>> VmCallArg for GuestStdUnorderedMap<K, V, H> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        pass_by_value(self, m, regs)
    }
}
