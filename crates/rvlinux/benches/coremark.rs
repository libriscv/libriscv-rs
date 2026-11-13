//! CoreMark 1.0: the Rust `rvlinux` (interpreter and JIT) against the C++
//! emulators, scored by CoreMark itself.
//!
//! ```sh
//! cargo bench -p rvlinux --bench coremark                      # interpreter
//! cargo bench -p rvlinux --features jit --bench coremark       # + JIT
//! cargo bench -p rvlinux --features aot-dylib --bench coremark # + AOT (translated to Rust)
//! COREMARK_RUNS=3 cargo bench ...                              # best of 3 (default 1)
//! COREMARK_ONLY="asmjit,Rust JIT" cargo bench ...              # variants whose name contains one
//! COREMARK_ELFS=../coremark/coremark-rv64gc,... cargo bench ... # extra prebuilt columns
//! ```
//!
//! Each column is a CoreMark 1.0 executable: the POSIX port linked
//! statically against glibc (`testutil::build_coremark_timed`), with
//! `ITERATIONS=0`, so CoreMark calibrates itself to at least 10 seconds of
//! real time (`clock_gettime`/`gettimeofday`) and reports Iterations/Sec,
//! which is the score printed here (best of `COREMARK_RUNS`). Every run must
//! print "Correct operation validated". Each run takes 10+ seconds, so the
//! instruction limit is raised (`--fuel`): the default 30G instructions stop
//! a JIT before then.
//!
//! The JITs compile up front: compiled in the background, the code can
//! arrive after CoreMark has calibrated on the interpreter, which then picks
//! too few iterations and fails the 10-second rule. The AOT column first
//! translates each ELF and compiles the translation with `rustc` into a
//! shared object (`--aot-save-shared`, not timed), then runs it with
//! `--aot-load`.
//!
//! Needs the RISC-V cross compiler and the CoreMark sources (`../coremark` or
//! `$COREMARK_DIR`). C++ emulators are used where they exist: the reference
//! (`$LIBRISCV_CPP_EMULATOR`), the dispatch variants from
//! `scripts/build-cpp-reference.sh`, and its `asmjit` build (falling back to
//! `$LIBRISCV_CPP/emulator/rvlinux-asmjit`).
//!
//! The difftests use different CoreMark builds (fixed iteration counts,
//! `rdinstret` as the clock), whose output is deterministic.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use testutil::CppDispatch;

/// The built columns: `(march, abi)`. Static glibc only exists for `lp64d`.
const BUILDS: &[(&str, &str)] = &[
    ("rv64imafdc_zicsr", "lp64d"),
    ("rv64imafdc_zba_zbb_zbs_zicsr", "lp64d"),
];

/// Enough instructions for 10+ seconds of JIT-compiled CoreMark.
const FUEL: &str = "10000000000000";

struct Variant {
    name: String,
    emulator: PathBuf,
    flags: Vec<&'static str>,
    /// Prepares an ELF for this variant (untimed); returns extra flags.
    prepare: Option<fn(&Path) -> Vec<String>>,
}

/// Translate `elf` to Rust and compile it into a shared object next to it
/// (`--aot-save-shared`; again whenever `rvlinux` is newer, since another
/// translator's code would be ignored): the flags that load it.
fn prepare_aot(elf: &Path) -> Vec<String> {
    let rust = PathBuf::from(env!("CARGO_BIN_EXE_rvlinux"));
    let mut so = elf.as_os_str().to_owned();
    so.push(".aot.so");
    let so = PathBuf::from(so);
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if modified(&so) < modified(&rust) {
        let status = Command::new(&rust)
            .arg("-Q")
            .arg("--aot-save-shared")
            .arg(&so)
            .arg(elf)
            .status()
            .unwrap_or_else(|e| panic!("{}: {e}", rust.display()));
        assert!(status.success(), "translating {}", elf.display());
    }
    vec!["--aot-load".into(), so.to_string_lossy().into_owned()]
}

fn variants() -> Vec<Variant> {
    let rust = PathBuf::from(env!("CARGO_BIN_EXE_rvlinux"));
    let mut v = vec![Variant {
        name: "Rust interpreter".into(),
        emulator: rust.clone(),
        flags: vec!["--accurate", "--fuel", FUEL],
        prepare: None,
    }];
    if cfg!(feature = "jit") {
        v.push(Variant {
            name: "Rust JIT".into(),
            emulator: rust.clone(),
            flags: vec!["--accurate", "--fuel", FUEL, "--jit", "--no-background"],
            prepare: None,
        });
    }
    if cfg!(feature = "aot-dylib") {
        v.push(Variant {
            name: "Rust AOT".into(),
            emulator: rust,
            flags: vec!["--accurate", "--fuel", FUEL],
            prepare: Some(prepare_aot),
        });
    }
    let mut cpp_interp = |name: &str, path: PathBuf| {
        if path.exists() {
            v.push(Variant {
                name: name.into(),
                emulator: path,
                flags: vec!["--no-translate", "--accurate", "--fuel", FUEL],
                prepare: None,
            });
        }
    };
    let threaded = testutil::reference_emulator_variant(CppDispatch::Threaded);
    if threaded.exists() {
        cpp_interp("C++ threaded", threaded);
    } else {
        cpp_interp("C++ reference", testutil::reference_emulator());
    }
    cpp_interp(
        "C++ tailcall",
        testutil::reference_emulator_variant(CppDispatch::Tailcall),
    );
    let mut asmjit = testutil::reference_asmjit_emulator();
    if !asmjit.exists() {
        asmjit = testutil::cpp_source_dir()
            .join("emulator")
            .join("rvlinux-asmjit");
    }
    if asmjit.exists() {
        v.push(Variant {
            name: "C++ asmjit".into(),
            emulator: asmjit,
            flags: vec!["--accurate", "--fuel", FUEL, "--no-background"],
            prepare: None,
        });
    }
    v
}

/// One run: CoreMark's Iterations/Sec, or `None` (with the output printed)
/// if the run didn't validate.
fn run(variant: &Variant, elf: &Path) -> Option<f64> {
    let extra = variant.prepare.map(|p| p(elf)).unwrap_or_default();
    let out = Command::new(&variant.emulator)
        .args(&variant.flags)
        .args(&extra)
        .arg(elf)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", variant.emulator.display()));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let score = stdout
        .lines()
        .find_map(|l| l.strip_prefix("Iterations/Sec"))
        .and_then(|l| l.trim_start_matches([' ', ':']).trim().parse().ok());
    match score {
        Some(score) if stdout.contains("Correct operation validated.") => Some(score),
        _ => {
            eprintln!(
                "{} failed on {}:\n{stdout}{}",
                variant.name,
                elf.display(),
                String::from_utf8_lossy(&out.stderr)
            );
            None
        }
    }
}

fn main() {
    if !testutil::need_compiler() || !testutil::need_coremark() {
        return;
    }
    let runs = std::env::var("COREMARK_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1u32)
        .max(1);
    let mut variants = variants();
    if let Ok(only) = std::env::var("COREMARK_ONLY") {
        variants.retain(|v| only.split(',').any(|f| v.name.contains(f.trim())));
    }

    let mut columns: Vec<(String, PathBuf)> = BUILDS
        .iter()
        .map(|&(march, abi)| {
            let elf = testutil::build_coremark_timed(march, abi).unwrap_or_else(|e| panic!("{e}"));
            (march.trim_end_matches("_zicsr").to_owned(), elf)
        })
        .collect();
    if let Ok(elfs) = std::env::var("COREMARK_ELFS") {
        for elf in elfs.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let elf = PathBuf::from(elf);
            let name = elf.file_name().map_or_else(
                || elf.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            columns.push((name, elf));
        }
    }

    println!("CoreMark 1.0, Iterations/Sec, best of {runs}\n");
    let mut header = format!("| {:<18} |", "");
    for (name, _) in &columns {
        header += &format!(" {name:>28} |");
    }
    println!("{header}");
    println!(
        "|{}|{}",
        "-".repeat(20),
        format!("{}:|", "-".repeat(29)).repeat(columns.len())
    );
    let mut failed = false;
    for variant in &variants {
        let mut line = format!("| {:<18} |", variant.name);
        for (_, elf) in &columns {
            let mut best: Option<f64> = None;
            for _ in 0..runs {
                match run(variant, elf) {
                    Some(score) => best = Some(best.map_or(score, |b| b.max(score))),
                    None => {
                        best = None;
                        break;
                    }
                }
            }
            match best {
                Some(score) => line += &format!(" {score:>28.1} |"),
                None => {
                    failed = true;
                    line += &format!(" {:>28} |", "FAILED");
                }
            }
        }
        println!("{line}");
    }
    assert!(!failed, "some runs failed to validate (output above)");
}
