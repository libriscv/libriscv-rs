//! Port of `guest/guest_cpp_hash.hpp`: the guest's `std::hash` (libstdc++'s
//! `_Hash_bytes`, 64-bit only), `std::equal_to`, and `_Prime_rehash_policy`'s
//! bucket counts.

use super::{GuestStdString, GuestType, MAX_BYTES};
use crate::error::Result;
use crate::machine::Machine;

/// libstdc++'s seed for all of its byte hashing.
pub const SEED: u64 = 0xc70f_6907;

fn shift_mix(v: u64) -> u64 {
    v ^ (v >> 47)
}

/// libstdc++'s `std::_Hash_bytes` for a 64-bit `size_t` (a MurmurHash2
/// variant): what `std::hash` uses for strings and floating-point values
/// (C++ `GuestStdHashBytes<RISCV64>::hash`).
pub fn hash_bytes(data: &[u8], seed: u64) -> u64 {
    const MUL: u64 = (0xc6a4_a793 << 32) + 0x5bd1_e995;
    let len = data.len() as u64;
    let mut hash = seed ^ len.wrapping_mul(MUL);
    let (chunks, rest) = data.as_chunks::<8>();
    for &chunk in chunks {
        let data = shift_mix(u64::from_le_bytes(chunk).wrapping_mul(MUL)).wrapping_mul(MUL);
        hash = (hash ^ data).wrapping_mul(MUL);
    }
    if !rest.is_empty() {
        // load_bytes: the remaining bytes, the last one most significant.
        let data = rest
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8).wrapping_add(u64::from(b)));
        hash = (hash ^ data).wrapping_mul(MUL);
    }
    hash = shift_mix(hash).wrapping_mul(MUL);
    shift_mix(hash)
}

/// A map key type (C++ `GuestStdHash<W, K>`): the guest's `std::hash<K>`
/// and `std::equal_to<K>`.
pub trait GuestHashKey: GuestType {
    /// libstdc++ stores the hash code in every node when `std::hash<K>`
    /// isn't considered fast (`__is_fast_hash`), which changes the node
    /// layout. True for `std::string`.
    const CACHE_HASH_CODE: bool;
    /// `std::hash<K>` of the key `self` (in guest memory, or a host copy
    /// of it).
    fn guest_hash(&self, m: &Machine) -> Result<u64>;
    /// `std::equal_to<K>`.
    fn guest_eq(&self, m: &Machine, other: &Self) -> Result<bool>;
}

/// Something a key of type `K` can be looked up with: the key type itself,
/// or a host value of it (a `&str` for a `std::string` key).
pub trait GuestKeyArg<K: GuestHashKey> {
    /// The guest's `std::hash<K>` of this key.
    fn key_hash(&self, m: &Machine) -> Result<u64>;
    /// True if `key` (a key of the map) equals this one.
    fn key_eq(&self, m: &Machine, key: &K) -> Result<bool>;
}

impl<K: GuestHashKey> GuestKeyArg<K> for K {
    fn key_hash(&self, m: &Machine) -> Result<u64> {
        self.guest_hash(m)
    }
    fn key_eq(&self, m: &Machine, key: &K) -> Result<bool> {
        key.guest_eq(m, self)
    }
}

macro_rules! int_keys {
    ($($t:ty),*) => {$(
        /// `std::hash` of an integer is the identity, converted to the
        /// guest's `size_t` (sign-extended).
        impl GuestHashKey for $t {
            const CACHE_HASH_CODE: bool = false;
            fn guest_hash(&self, _m: &Machine) -> Result<u64> {
                #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
                Ok(*self as u64)
            }
            fn guest_eq(&self, _m: &Machine, other: &Self) -> Result<bool> {
                Ok(self == other)
            }
        }
    )*};
}
int_keys!(i8, i16, i32, i64, u8, u16, u32, u64);

impl GuestHashKey for bool {
    const CACHE_HASH_CODE: bool = false;
    fn guest_hash(&self, _m: &Machine) -> Result<u64> {
        Ok(u64::from(*self))
    }
    fn guest_eq(&self, _m: &Machine, other: &Self) -> Result<bool> {
        Ok(self == other)
    }
}

macro_rules! float_keys {
    ($($t:ty),*) => {$(
        /// `std::hash` of a float hashes its bytes, except that 0.0 and
        /// -0.0 both hash to zero.
        impl GuestHashKey for $t {
            const CACHE_HASH_CODE: bool = false;
            fn guest_hash(&self, _m: &Machine) -> Result<u64> {
                Ok(if *self != 0.0 { hash_bytes(&self.to_le_bytes(), SEED) } else { 0 })
            }
            fn guest_eq(&self, _m: &Machine, other: &Self) -> Result<bool> {
                Ok(self == other)
            }
        }
    )*};
}
float_keys!(f32, f64);

/// `std::hash<std::string>` isn't a "fast" hash, so libstdc++ stores the
/// hash code of every key in its node.
impl GuestHashKey for GuestStdString {
    const CACHE_HASH_CODE: bool = true;
    fn guest_hash(&self, m: &Machine) -> Result<u64> {
        Ok(hash_bytes(self.to_view(m, MAX_BYTES)?, SEED))
    }
    fn guest_eq(&self, m: &Machine, other: &Self) -> Result<bool> {
        Ok(self.to_view(m, MAX_BYTES)? == other.to_view(m, MAX_BYTES)?)
    }
}

impl GuestKeyArg<GuestStdString> for [u8] {
    fn key_hash(&self, _m: &Machine) -> Result<u64> {
        Ok(hash_bytes(self, SEED))
    }
    fn key_eq(&self, m: &Machine, key: &GuestStdString) -> Result<bool> {
        Ok(key.to_view(m, MAX_BYTES)? == self)
    }
}
impl GuestKeyArg<GuestStdString> for str {
    fn key_hash(&self, m: &Machine) -> Result<u64> {
        self.as_bytes().key_hash(m)
    }
    fn key_eq(&self, m: &Machine, key: &GuestStdString) -> Result<bool> {
        self.as_bytes().key_eq(m, key)
    }
}
impl GuestKeyArg<GuestStdString> for &str {
    fn key_hash(&self, m: &Machine) -> Result<u64> {
        self.as_bytes().key_hash(m)
    }
    fn key_eq(&self, m: &Machine, key: &GuestStdString) -> Result<bool> {
        self.as_bytes().key_eq(m, key)
    }
}
impl GuestKeyArg<GuestStdString> for String {
    fn key_hash(&self, m: &Machine) -> Result<u64> {
        self.as_bytes().key_hash(m)
    }
    fn key_eq(&self, m: &Machine, key: &GuestStdString) -> Result<bool> {
        self.as_bytes().key_eq(m, key)
    }
}

/// Whether a map stores hash codes in its nodes, which depends on its hash
/// function (C++'s `CacheHashCode` argument of `GuestStdUnorderedMap`).
pub trait HashPolicy<K> {
    const CACHE_HASH_CODE: bool;
}

/// The map hashes with `std::hash<K>`: hash codes are stored for the key
/// types libstdc++ considers slow to hash (`std::string`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StdHash;
impl<K: GuestHashKey> HashPolicy<K> for StdHash {
    const CACHE_HASH_CODE: bool = K::CACHE_HASH_CODE;
}

/// The map hashes with a custom hash function (which must compute the
/// same values as `std::hash<K>`): libstdc++ considers every hash function
/// fast except its own slow ones, so no hash codes are stored (C++
/// `GuestStdUnorderedMap<W, K, V, false>`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CustomHash;
impl<K> HashPolicy<K> for CustomHash {
    const CACHE_HASH_CODE: bool = false;
}

/// The bucket counts of libstdc++'s `_Prime_rehash_policy`. The list is
/// truncated: beyond the last entry the bucket array stops growing.
pub(crate) const FAST_BKT: [u8; 14] = [2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13];
pub(crate) const PRIMES: [u32; 186] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    103, 109, 113, 127, 137, 139, 149, 157, 167, 179, 193, 199, 211, 227, 241, 257, 277, 293, 313,
    337, 359, 383, 409, 439, 467, 503, 541, 577, 619, 661, 709, 761, 823, 887, 953, 1031, 1109,
    1193, 1289, 1381, 1493, 1613, 1741, 1879, 2029, 2179, 2357, 2549, 2753, 2971, 3209, 3469, 3739,
    4027, 4349, 4703, 5087, 5503, 5953, 6427, 6949, 7517, 8123, 8783, 9497, 10273, 11113, 12011,
    12983, 14033, 15173, 16411, 17749, 19183, 20753, 22447, 24281, 26267, 28411, 30727, 33223,
    35933, 38873, 42043, 45481, 49201, 53201, 57557, 62233, 67307, 72817, 78779, 85229, 92203,
    99733, 107897, 116731, 126271, 136607, 147793, 159871, 172933, 187091, 202409, 218971, 236897,
    256279, 277261, 299951, 324503, 351061, 379787, 410857, 444487, 480881, 520241, 562841, 608903,
    658753, 712697, 771049, 834181, 902483, 976369, 1056323, 1142821, 1236397, 1337629, 1447153,
    1565659, 1693859, 1832561, 1982627, 2144977, 2320627, 2510653, 2716249, 2938679, 3179303,
    3439651, 3721303, 4026031, 4355707, 4712381, 5098259, 5515729, 5967347, 6456007, 6984629,
    7556579, 8175383, 8844859, 9569143, 10352717, 11200489, 12117689, 13109983, 14183539, 15345007,
    16601593, 17961079,
];
