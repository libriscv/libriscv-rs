//! The JIT code cache through the library API: a [`JitCacheOutput`] with
//! several segments, saved as a shared object, loaded into the
//! process-wide registry and run by machines without the JIT. In its own
//! test binary, since the registry is process-wide (and on by default).

#![cfg(feature = "jit")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::process::Command;

use libriscv::jit_cache::{self, JitCacheOutput};
use libriscv::{Machine, MachineOptions};

const A: u64 = 0x10000;
const B: u64 = 0x11000;

const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13
}
const fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
    (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33
}
const fn bne(rs1: u32, rs2: u32, off: i32) -> u32 {
    let o = off as u32;
    ((o >> 12 & 1) << 31)
        | ((o >> 5 & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (1 << 12)
        | ((o >> 1 & 0xF) << 8)
        | ((o >> 11 & 1) << 7)
        | 0x63
}
const fn jal(rd: u32, off: i32) -> u32 {
    let o = off as u32;
    ((o >> 20 & 1) << 31)
        | ((o >> 1 & 0x3FF) << 21)
        | ((o >> 11 & 1) << 20)
        | ((o >> 12 & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Segment A loops, then calls into segment B, which loops and returns;
/// then A spins.
fn segments() -> (Vec<u8>, Vec<u8>) {
    let a = bytes(&[
        addi(5, 0, 500),
        // loop:
        addi(10, 10, 3),
        add(11, 11, 10),
        addi(5, 5, -1),
        bne(5, 0, -12),
        // lui t2, B; jalr ra, 0(t2) (a JAL out of the segment is invalid)
        ((B as u32 >> 12) << 12) | (7 << 7) | 0x37,
        (7 << 15) | (1 << 7) | 0x67,
        addi(12, 12, 1),
        jal(0, 0),
    ]);
    let b = bytes(&[
        addi(6, 0, 300),
        // loop:
        add(13, 13, 11),
        addi(6, 6, -1),
        bne(6, 0, -8),
        // ret
        (1 << 15) | 0x67,
    ]);
    (a, b)
}

/// The options of most machines here: a 1 MiB arena.
fn small() -> MachineOptions {
    MachineOptions {
        memory_max: 1 << 20,
        ..Default::default()
    }
}

fn machine(options: MachineOptions) -> Machine {
    let mut m = Machine::new(&[], options).unwrap();
    let (a, b) = segments();
    m.init_execute_area(&b, B).unwrap();
    m.init_execute_area(&a, A).unwrap();
    m.cpu.jump(A).unwrap();
    m
}

fn state(m: &mut Machine) -> Vec<u64> {
    let result = m.try_simulate(10_000, 0);
    assert!(result.is_ok(), "{result:?}");
    let mut s = m.cpu.registers().all().to_vec();
    s.extend([m.cpu.pc(), m.instruction_counter()]);
    s
}

fn cached(m: &Machine) -> [bool; 2] {
    [A, B].map(|v| m.memory.exec_segment_for(v).unwrap().jit_is_cached())
}

#[test]
fn saved_segments_run_like_the_interpreter() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let have_cc = Command::new(&cc)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !testutil::tool_available(&format!("host `{cc}`"), have_cc) {
        return;
    }
    let expected = state(&mut machine(MachineOptions {
        jit_cache: false,
        ..small()
    }));

    let output = JitCacheOutput::new();
    let mut m = machine(MachineOptions {
        jit: true,
        jit_cache_output: Some(output.clone()),
        ..small()
    });
    assert_eq!(state(&mut m), expected);
    assert_eq!(output.len(), 2);
    assert!(output.object().unwrap().starts_with(b"\x7FELF"));

    let dir = testutil::workspace_root()
        .join("target")
        .join("jit-cache-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let so = dir.join("library-api.so");
    output.write_shared_object(&so).unwrap();
    assert_eq!(jit_cache::registered_segments(), 0);
    assert_eq!(jit_cache::load_shared_object(&so).unwrap(), 2);
    // Loading it again adds nothing: the segments have code already.
    assert_eq!(jit_cache::load_shared_object(&so).unwrap(), 0);
    assert_eq!(jit_cache::registered_segments(), 2);

    // Without the JIT, with and without the cache.
    let mut m = machine(small());
    assert_eq!(cached(&m), [true, true]);
    assert_eq!(state(&mut m), expected);
    let m = machine(MachineOptions {
        jit_cache: false,
        ..small()
    });
    assert_eq!(cached(&m), [false, false]);
    // Another memory layout: the code isn't used.
    let mut m = machine(MachineOptions {
        memory_max: 2 << 20,
        ..small()
    });
    assert_eq!(cached(&m), [false, false]);
    assert_eq!(state(&mut m), expected);
}
