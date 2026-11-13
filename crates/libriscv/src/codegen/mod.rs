//! Port of `asmjit/aj_decode.hpp` and `asmjit/aj_translate.cpp`: what the
//! native-code backends share. Instruction selection ([`classify`]) and
//! region formation ([`region`]) decide which instructions run natively
//! and how they are grouped into functions; the Cranelift JIT (`jit`)
//! turns each region into machine code, the ahead-of-time translator
//! (`aot`) into a Rust function.

pub(crate) mod classify;
pub(crate) mod region;

use crate::native_code::MemoryLayout;

/// 64-bit FNV-1a, a word at a time. Not cryptographic: it tells segments
/// apart, it doesn't defend against crafted collisions (native code
/// from a file is trusted anyway).
pub(crate) fn hash64(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let (words, rest) = bytes.as_chunks::<8>();
    for &w in words {
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(PRIME);
    }
    for &b in rest {
        h = (h ^ u64::from(b)).wrapping_mul(PRIME);
    }
    (h ^ bytes.len() as u64).wrapping_mul(PRIME)
}

/// What the native code built for a segment depends on: the backend's
/// build (`build`, a fingerprint of its version and source), the segment
/// and the guest memory layout baked into the code. The key of the JIT
/// cache (`jit::cache::CacheKey`) and of translations (`aot::AotKey`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct NativeKey {
    pub build: u64,
    pub vaddr: u64,
    pub len: u64,
    pub crc32c: u32,
    /// [`hash64`] of the code.
    pub hash: u64,
    pub execute_only: bool,
    /// The libc fast path patched the segment.
    pub libc_fastpath: bool,
    pub layout: MemoryLayout,
}

impl NativeKey {
    /// The key of the segment `code` at `vaddr` (whose CRC32-C is
    /// `crc32c`), for a backend whose build is still to be set
    /// ([`with_build`](Self::with_build)). Hashes the code: make one per
    /// segment and derive the backends' keys from it.
    pub(crate) fn new(
        code: &[u8],
        vaddr: u64,
        crc32c: u32,
        execute_only: bool,
        libc_fastpath: bool,
        layout: MemoryLayout,
    ) -> Self {
        Self {
            build: 0,
            vaddr,
            len: code.len() as u64,
            crc32c,
            hash: hash64(code),
            execute_only,
            libc_fastpath,
            layout,
        }
    }

    pub(crate) const fn with_build(self, build: u64) -> Self {
        Self { build, ..self }
    }

    /// The number of [`words`](Self::words).
    pub(crate) const WORDS: usize = 11;

    /// The key as `u64`s, as metadata stores it.
    pub(crate) fn words(&self) -> [u64; Self::WORDS] {
        let flags = u64::from(self.execute_only) | u64::from(self.libc_fastpath) << 1;
        let l = &self.layout;
        [
            self.build,
            self.vaddr,
            self.len,
            u64::from(self.crc32c),
            self.hash,
            flags,
            l.mask,
            l.read.0,
            l.read.1,
            l.write.0,
            l.write.1,
        ]
    }

    /// The key of [`words`](Self::words), or `None` if they aren't one.
    pub(crate) fn from_words(w: [u64; Self::WORDS]) -> Option<Self> {
        let [build, vaddr, len, crc, hash, flags, mask, rb, rl, wb, wl] = w;
        Some(Self {
            build,
            vaddr,
            len,
            crc32c: u32::try_from(crc).ok()?,
            hash,
            execute_only: flags & 1 != 0,
            libc_fastpath: flags & 2 != 0,
            layout: MemoryLayout {
                mask,
                read: (rb, rl),
                write: (wb, wl),
            },
        })
    }
}

/// The source of what decides the native code built for a segment, part
/// of both backends' fingerprints: which instructions run natively and
/// how they are grouped, the decoding and decoder layout they are built
/// from, and the entries the libc fast path patches (native code leaves
/// those alone). A change to any of them may change what a segment
/// compiles to, also without a version bump.
pub(crate) const SHARED_SOURCES: &[&str] = &[
    include_str!("mod.rs"),
    include_str!("region.rs"),
    include_str!("classify.rs"),
    include_str!("../native_code.rs"),
    include_str!("../decoder_cache.rs"),
    include_str!("../decode.rs"),
    include_str!("../bytecode.rs"),
    include_str!("../instruction.rs"),
    include_str!("../native/libc.rs"),
];

/// A fingerprint of `text` followed by the hashes of `SHARED_SOURCES` and
/// of the backend's own `sources`.
pub(crate) fn fingerprint(mut text: String, sources: &[&str]) -> u64 {
    for source in SHARED_SOURCES.iter().chain(sources) {
        text.push_str(&format!(" {:016x}", hash64(source.as_bytes())));
    }
    hash64(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_tells_lengths_and_contents_apart() {
        assert_ne!(hash64(b""), hash64(b"\0"));
        assert_ne!(hash64(b"abcdefgh"), hash64(b"abcdefgi"));
        assert_ne!(hash64(&[0; 8]), hash64(&[0; 16]));
    }

    #[test]
    fn native_key_round_trips() {
        let key = NativeKey {
            build: 1,
            vaddr: 0x10000,
            len: 64,
            crc32c: 0xDEAD_BEEF,
            hash: 7,
            execute_only: true,
            libc_fastpath: true,
            layout: MemoryLayout {
                mask: 0,
                read: (0x1000, 0x2000),
                write: (0x3000, 0x400),
            },
        };
        assert_eq!(NativeKey::from_words(key.words()), Some(key));
        assert_eq!(key.with_build(9).build, 9);
        let mut w = key.words();
        w[3] = 1 << 32;
        assert_eq!(NativeKey::from_words(w), None);
    }
}
