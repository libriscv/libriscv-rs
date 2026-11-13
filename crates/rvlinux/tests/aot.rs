//! Ahead-of-time translation to Rust: `--aot-save`, `--aot-save-shared`,
//! `--aot-load` and translations compiled into `rvlinux`
//! (`RVLINUX_AOT_EMBED`). Translated code must behave exactly like the
//! interpreter: the same output, exit code and instruction count, and it
//! must only be used for the segments and memory layout it was made for.

#![cfg(feature = "aot-dylib")]
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

/// The cross compiler, and `rustc` to compile translations.
fn have_tools() -> bool {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let ok = Command::new(&rustc)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    testutil::need_compiler() && testutil::tool_available(&format!("`{rustc}`"), ok)
}

/// A fresh scratch file for `test`, in the target directory.
fn scratch(test: &str, name: &str) -> PathBuf {
    let dir = testutil::workspace_root()
        .join("target")
        .join("aot-tests")
        .join("rvlinux")
        .join(test);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn run(elf: &Path, flags: &[&str]) -> RunOutput {
    let mut all = vec!["-a"];
    all.extend_from_slice(flags);
    let out = testutil::run_rust_with(rvlinux(), elf, &all, &["arg"]).unwrap();
    assert!(out.exit_code.is_some(), "{flags:?}: {out:?}");
    assert!(out.instructions.is_some(), "{flags:?}: {out:?}");
    out
}

/// stderr of a verbose run (`-v`): the loader's messages about
/// translations.
fn messages(elf: &Path, flags: &[&str]) -> String {
    let mut all = vec!["-v"];
    all.extend_from_slice(flags);
    run(elf, &all)
        .stderr
        .lines()
        .filter(|l| l.contains("translation for the segment"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Translate `elf` into a shared object and check that loading it gives
/// the interpreter's results, and only where it applies.
fn check(elf: &Path, test: &str) {
    let expected = run(elf, &[]);
    let so = scratch(test, "translation.so");
    let rs = scratch(test, "translation.rs");
    let (so_s, rs_s) = (so.to_str().unwrap(), rs.to_str().unwrap());
    // Translating doesn't change the run.
    let saved = run(elf, &["--aot-save-shared", so_s, "--aot-save", rs_s]);
    assert_eq!(saved, expected);
    assert!(so.exists() && rs.exists());

    // Loading: the translation is used, for every segment.
    assert_eq!(run(elf, &["--aot-load", so_s]), expected);
    let found = messages(elf, &["--aot-load", so_s]);
    assert!(found.contains("Found translation"), "{found}");
    assert!(!found.contains("No translation"), "{found}");
    // ... also together with the JIT, which then compiles nothing.
    if cfg!(feature = "jit") {
        assert_eq!(run(elf, &["--jit", "--aot-load", so_s]), expected);
    }
    // Ignored with --no-aot.
    let none = messages(elf, &["--aot-load", so_s, "--no-aot"]);
    assert!(none.is_empty(), "{none}");
    // Not used for another memory layout (a smaller arena), which the
    // code has baked in.
    let flags = ["--aot-load", so_s, "--memory", "256"];
    assert_eq!(run(elf, &flags), expected);
    let found = messages(elf, &flags);
    assert!(found.contains("No translation"), "{found}");
    assert!(!found.contains("Found"), "{found}");
}

/// CoreMark, freestanding: quick to translate and compile.
#[test]
fn freestanding_coremark() {
    if !have_tools() || !testutil::need_coremark() {
        return;
    }
    let elf =
        testutil::build_coremark("rv64imafdc_zicsr", "lp64d", 10).unwrap_or_else(|e| panic!("{e}"));
    check(&elf, "coremark");
}

/// A static glibc executable: libc's ~300 KB of code become ~15 MB of Rust,
/// which `rustc` takes a minute to compile.
#[test]
#[ignore = "compiles a translation of static glibc (~1 min)"]
fn static_executable() {
    if !have_tools() {
        return;
    }
    let program = r#"
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
    let elf = testutil::build(program, "-O2 -static", testutil::Lang::C)
        .unwrap_or_else(|e| panic!("{e}"));
    check(&elf, "static");
}

/// A missing file, or one without a translation, is an error, not a crash.
#[test]
fn loading_a_bad_file_fails() {
    if !testutil::need_compiler() || !testutil::need_coremark() {
        return;
    }
    let elf =
        testutil::build_coremark("rv64imafdc_zicsr", "lp64d", 10).unwrap_or_else(|e| panic!("{e}"));
    let missing = scratch("bad", "missing.so");
    let not_ours = scratch("bad", "not-ours.so");
    std::fs::write(&not_ours, b"not a shared object").unwrap();
    for file in [missing, not_ours] {
        let out = Command::new(rvlinux())
            .arg("--aot-load")
            .arg(&file)
            .arg(&elf)
            .output()
            .unwrap();
        assert!(!out.status.success(), "{out:?}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.starts_with("Could not load"), "{stderr}");
    }
}

/// A translation compiled into `rvlinux` is registered at startup. Builds
/// a second `rvlinux` with only the `aot` feature (no unsafe code in any
/// crate of the build), in its own target directory, `target/aot-embed`,
/// so that it doesn't rebuild or block this one.
#[test]
#[ignore = "builds rvlinux again, in target/aot-embed"]
fn embedded_translation() {
    if !have_tools() || !testutil::need_coremark() {
        return;
    }
    let elf =
        testutil::build_coremark("rv64imafdc_zicsr", "lp64d", 10).unwrap_or_else(|e| panic!("{e}"));
    let rs = scratch("embedded", "translation.rs");
    let expected = run(&elf, &["--aot-save", rs.to_str().unwrap()]);
    let root = testutil::workspace_root();
    let target = root.join("target").join("aot-embed");
    let status = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args(["build", "-p", "rvlinux", "--features", "aot"])
        .env("CARGO_TARGET_DIR", &target)
        .env("RVLINUX_AOT_EMBED", &rs)
        .status()
        .unwrap();
    assert!(status.success());
    let embedded = target.join("debug").join("rvlinux");
    let out = testutil::run_rust_with(&embedded, &elf, &["-a"], &["arg"]).unwrap();
    assert_eq!(out, expected);
    let out = testutil::run_rust_with(&embedded, &elf, &["-a", "-v"], &["arg"]).unwrap();
    assert!(out.stderr.contains("Found translation"), "{}", out.stderr);
    // Without RVLINUX_AOT_EMBED there's no `aot_embed` module.
    assert!(!out.stderr.contains("No translation"), "{}", out.stderr);
}
