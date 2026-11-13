//! The JIT code cache: `--jit-save-object`, `--jit-save-shared`,
//! `--jit-load` and code linked into `rvlinux` (`RVLINUX_JIT_EMBED`).
//! Saved code must behave exactly like the JIT (and so the interpreter):
//! the same output, exit code and instruction count, and it must only be
//! used for the segments and memory layout it was compiled for.

#![cfg(feature = "jit")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use testutil::RunOutput;

fn rvlinux() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_rvlinux"))
}

const PROGRAM: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned collatz(unsigned long n) {
    unsigned steps = 0;
    while (n != 1) { n = (n & 1) ? 3 * n + 1 : n / 2; steps++; }
    return steps;
}
int main(int argc, char** argv) {
    unsigned best = 0, arg = 0;
    for (unsigned i = 1; i < 20000; i++) {
        unsigned s = collatz(i);
        if (s > best) { best = s; arg = i; }
    }
    char* buf = malloc(4096);
    snprintf(buf, 4096, "collatz %u %u %d", arg, best, argc);
    double x = 0;
    for (int i = 1; i < 1000; i++) x += 1.0 / (i * (double)i);
    printf("%s %.9f %zu\n", buf, x, strlen(buf));
    free(buf);
    return best & 0x7F;
}
"#;

/// Tools for these tests: the cross compiler, and a host C compiler driver
/// to link shared objects.
fn have_tools() -> bool {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let host_cc = Command::new(&cc)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    testutil::need_compiler() && testutil::tool_available(&format!("host `{cc}`"), host_cc)
}

fn build(args: &str) -> PathBuf {
    testutil::build(PROGRAM, args, testutil::Lang::C).unwrap_or_else(|e| panic!("{e}"))
}

/// A fresh scratch file for `test`, in the target directory.
fn scratch(test: &str, name: &str) -> PathBuf {
    let dir = testutil::workspace_root()
        .join("target")
        .join("jit-cache-tests")
        .join(test);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn run(elf: &Path, flags: &[&str]) -> RunOutput {
    let out = testutil::run_rust_with(rvlinux(), elf, flags, &["arg"]).unwrap();
    assert!(out.exit_code.is_some(), "{flags:?}: {out:?}");
    out
}

/// stderr of a verbose run (`-v`): the cache's messages.
fn cache_messages(elf: &Path, flags: &[&str]) -> String {
    let mut all = vec!["-v"];
    all.extend_from_slice(flags);
    run(elf, &all)
        .stderr
        .lines()
        .filter(|l| l.contains("cached JIT code"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn check(elf: &Path, test: &str) {
    let expected = run(elf, &[]);
    let so = scratch(test, "code.so");
    let o = scratch(test, "code.o");
    let (so_s, o_s) = (so.to_str().unwrap(), o.to_str().unwrap());
    // Saving runs the JIT: the same results.
    let saved = run(elf, &["--jit-save-shared", so_s, "--jit-save-object", o_s]);
    assert_eq!(saved, expected);
    assert!(so.exists() && o.exists());

    // Loading: the saved code is used, for every segment.
    let loaded = run(elf, &["--jit-load", so_s]);
    assert_eq!(loaded, expected);
    let messages = cache_messages(elf, &["--jit-load", so_s]);
    assert!(messages.contains("Found cached JIT code"), "{messages}");
    assert!(!messages.contains("No cached JIT code"), "{messages}");
    // ... also together with the JIT, which then compiles nothing.
    assert_eq!(run(elf, &["--jit", "--jit-load", so_s]), expected);

    // Ignored with --no-jit-cache.
    let messages = cache_messages(elf, &["--jit-load", so_s, "--no-jit-cache"]);
    assert!(messages.is_empty(), "{messages}");
    // Not used for another memory layout (a smaller arena), which the code
    // has baked in.
    let other = ["--jit-load", so_s, "--memory", "256"];
    assert_eq!(run(elf, &other), expected);
    let messages = cache_messages(elf, &other);
    assert!(!messages.contains("Found"), "{messages}");
    assert!(messages.contains("No cached JIT code"), "{messages}");
}

#[test]
fn static_executable() {
    if !have_tools() {
        return;
    }
    check(&build("-O2 -static"), "static");
}

/// `ld.so`, `libc.so` and the program are separate segments, created at
/// run time. Slow (~20 s): saving compiles all of `libc.so` up front (see
/// `mode_flags` in `difftest.rs`); `libriscv`'s `tests/jit_cache.rs`
/// covers files with several segments.
#[test]
#[ignore = "compiles all of libc.so (~20 s)"]
fn dynamic_executable() {
    if !have_tools() {
        return;
    }
    check(&build("-O2"), "dynamic");
}

/// `-Q` saves the code of the program's segment without running it.
#[test]
fn quit_after_loading_saves_code() {
    if !have_tools() {
        return;
    }
    let elf = build("-O2 -static");
    let so = scratch("quit", "code.so");
    let so_s = so.to_str().unwrap();
    let out = Command::new(rvlinux())
        .args(["-Q", "--jit-save-shared", so_s])
        .arg(&elf)
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).is_empty(), "{out:?}");
    assert_eq!(run(&elf, &["--jit-load", so_s]), run(&elf, &[]));
    let messages = cache_messages(&elf, &["--jit-load", so_s]);
    assert!(messages.contains("Found cached JIT code"), "{messages}");
}

/// A missing or foreign file is an error, not a crash.
#[test]
fn loading_a_bad_file_fails() {
    if !have_tools() {
        return;
    }
    let elf = build("-O2 -static");
    let missing = scratch("bad", "missing.so");
    let not_ours = scratch("bad", "not-ours.so");
    // A shared object without a table.
    let c = scratch("bad", "empty.c");
    std::fs::write(&c, "int nothing_here;\n").unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(cc)
        .args(["-shared", "-fPIC", "-o"])
        .arg(&not_ours)
        .arg(&c)
        .status()
        .unwrap();
    assert!(status.success());
    for file in [missing, not_ours] {
        let out = Command::new(rvlinux())
            .arg("--jit-load")
            .arg(&file)
            .arg(&elf)
            .output()
            .unwrap();
        assert!(!out.status.success(), "{out:?}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.starts_with("Could not load"), "{stderr}");
    }
}

/// Code linked into `rvlinux` is found by itself. Builds a second `rvlinux`
/// (in its own target directory, `target/jit-embed`, so that it doesn't
/// rebuild or block this one), which takes a while the first time.
#[test]
#[ignore = "builds rvlinux again, in target/jit-embed"]
fn embedded_object() {
    if !have_tools() {
        return;
    }
    let elf = build("-O2 -static");
    let o = scratch("embedded", "code.o");
    let o_s = o.to_str().unwrap();
    let expected = run(&elf, &["--jit-save-object", o_s]);
    let root = testutil::workspace_root();
    let target = root.join("target").join("jit-embed");
    let status = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args(["build", "-p", "rvlinux", "--features", "jit"])
        .env("CARGO_TARGET_DIR", &target)
        .env("RVLINUX_JIT_EMBED", &o)
        .status()
        .unwrap();
    assert!(status.success());
    let embedded = target.join("debug").join("rvlinux");
    let out = testutil::run_rust_with(&embedded, &elf, &[], &["arg"]).unwrap();
    assert_eq!(out, expected);
    let out = testutil::run_rust_with(&embedded, &elf, &["-v"], &["arg"]).unwrap();
    assert!(
        out.stderr.contains("Found cached JIT code"),
        "{}",
        out.stderr
    );
}
