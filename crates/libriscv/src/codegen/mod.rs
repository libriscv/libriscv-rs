//! Port of `asmjit/aj_decode.hpp` and `asmjit/aj_translate.cpp`: what the
//! native-code backends share. Instruction selection ([`classify`]) and
//! region formation ([`region`]) decide which instructions run natively
//! and how they are grouped into functions; the Cranelift JIT (`jit`)
//! turns each region into machine code, the ahead-of-time translator
//! (`aot`) into a Rust function.

pub(crate) mod classify;
pub(crate) mod region;

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
