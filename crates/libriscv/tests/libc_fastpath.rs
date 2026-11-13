//! The libc fast path (`MachineOptions::libc_fastpath`): port of
//! `tests/unit/libc_fastpath.cpp`. `rvlinux`'s difftests check the
//! instruction counts against the C++ emulator.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
// Every test runs a glibc guest, and the distribution's glibc is RV64GC.
#![cfg(all(feature = "ext-a", feature = "ext-c", feature = "ext-f"))]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use libriscv::{EbreakLocation, Machine, MachineOptions, SYSCALL_EBREAK_NR};

const MAX_MEMORY: u64 = 32 << 20;
const MAX_INSTRUCTIONS: u64 = 200_000_000;

fn exerciser() -> Option<Vec<u8>> {
    if !testutil::need_compiler() {
        return None;
    }
    Some(
        testutil::build_and_load(testutil::LIBC_FASTPATH_EXERCISER, "-O2 -static")
            .unwrap_or_else(|e| panic!("{e}")),
    )
}

#[derive(Default)]
struct State {
    output: Mutex<String>,
    ebreaks: AtomicUsize,
}

/// System call handlers and the printer are plain function pointers, so
/// the test state travels in the machine's userdata.
fn collect_output(m: &Machine, data: &[u8]) {
    let state = m.userdata::<State>().unwrap();
    state
        .output
        .lock()
        .unwrap()
        .push_str(&String::from_utf8_lossy(data));
}

fn take_output(machine: &mut Machine) -> String {
    let state = machine.take_userdata::<State>().unwrap();
    state.output.into_inner().unwrap()
}

fn run_exerciser(binary: &[u8], fastpath: bool) -> String {
    let options = MachineOptions {
        memory_max: MAX_MEMORY,
        libc_fastpath: fastpath,
        ..Default::default()
    };
    let mut machine = Machine::new(binary, options).unwrap();
    machine.set_userdata(State::default());
    machine.set_printer(collect_output);
    machine.setup_linux_syscalls(false, false);
    machine
        .setup_linux(&["libc_fastpath"], &["LC_ALL=C"])
        .unwrap();
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    take_output(&mut machine)
}

/// "Hot-patched libc functions behave like the real ones".
#[test]
fn hot_patched_functions_behave_like_the_real_ones() {
    let Some(binary) = exerciser() else {
        return;
    };
    // The unpatched run is the reference: it runs the guest's own libc.
    let reference = run_exerciser(&binary, false);
    assert_eq!(reference, "libc: 0 failures\n");
    let patched = run_exerciser(&binary, true);
    assert_eq!(patched, reference);
}

/// "Hot-patching leaves guest memory untouched".
#[test]
fn hot_patching_leaves_guest_memory_untouched() {
    let Some(binary) = exerciser() else {
        return;
    };
    let options = MachineOptions {
        memory_max: MAX_MEMORY,
        ..Default::default()
    };
    let plain = Machine::new(&binary, options.clone()).unwrap();
    let patched = Machine::new(
        &binary,
        MachineOptions {
            libc_fastpath: true,
            ..options
        },
    )
    .unwrap();
    let memcpy_addr = plain.address_of("memcpy").unwrap();
    assert_ne!(memcpy_addr, 0);
    // Only the decoder cache is rewritten, so the machine code a debugger,
    // a backtrace or the precise simulator reads is still the original.
    let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
    plain.copy_from_guest(&mut a, memcpy_addr).unwrap();
    patched.copy_from_guest(&mut b, memcpy_addr).unwrap();
    assert_eq!(a, b);
}

/// "Hot-patching does not break real breakpoints".
#[test]
fn hot_patching_does_not_break_real_breakpoints() {
    let Some(binary) = exerciser() else {
        return;
    };
    let options = MachineOptions {
        memory_max: MAX_MEMORY,
        ebreak_locations: vec![EbreakLocation::Symbol("main".into())],
        libc_fastpath: true,
        ..Default::default()
    };
    let mut machine = Machine::new(&binary, options).unwrap();
    machine.set_userdata(State::default());
    machine.set_printer(collect_output);
    machine.setup_linux_syscalls(false, false);
    machine
        .setup_linux(&["libc_fastpath"], &["LC_ALL=C"])
        .unwrap();
    // setup_linux_syscalls() installs its own EBREAK handler *after* the
    // fast path took over EBREAK, and must end up in the chain rather than
    // replacing it. Installing one here has to work the same way.
    machine.install_syscall_handler(SYSCALL_EBREAK_NR, |m| {
        m.userdata::<State>()
            .unwrap()
            .ebreaks
            .fetch_add(1, Ordering::Relaxed);
        // Resume at the (unmodified) instruction the breakpoint replaced.
        m.simulate_precise()
    });
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(
        machine
            .userdata::<State>()
            .unwrap()
            .ebreaks
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(take_output(&mut machine), "libc: 0 failures\n");
}

/// Breakpoints (`ebreak_locations`) without the libc fast path: an EBREAK
/// handler installed later replaces the old one as usual.
#[test]
fn breakpoints_without_the_fast_path() {
    let Some(binary) = exerciser() else {
        return;
    };
    let options = MachineOptions {
        memory_max: MAX_MEMORY,
        ebreak_locations: vec![EbreakLocation::Symbol("main".into())],
        ..Default::default()
    };
    let mut machine = Machine::new(&binary, options).unwrap();
    machine.set_userdata(State::default());
    machine.set_printer(collect_output);
    machine.setup_linux_syscalls(false, false);
    machine
        .setup_linux(&["libc_fastpath"], &["LC_ALL=C"])
        .unwrap();
    machine.install_syscall_handler(SYSCALL_EBREAK_NR, |m| {
        m.userdata::<State>()
            .unwrap()
            .ebreaks
            .fetch_add(1, Ordering::Relaxed);
        m.simulate_precise()
    });
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(
        machine
            .userdata::<State>()
            .unwrap()
            .ebreaks
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(take_output(&mut machine), "libc: 0 failures\n");
}
