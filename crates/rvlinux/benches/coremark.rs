//! CoreMark: the Rust `rvlinux` (interpreter and JIT) against the C++
//! emulators, by wall-clock time of the whole process (loading, compiling
//! and running).
//!
//! ```sh
//! cargo bench -p rvlinux --bench coremark                      # interpreter
//! cargo bench -p rvlinux --features jit --bench coremark       # + JIT
//! COREMARK_ITERATIONS=4000 COREMARK_RUNS=5 cargo bench ...     # defaults: 4000, 3
//! ```
//!
//! Needs the RISC-V cross compiler and the CoreMark sources (`../coremark` or
//! `$COREMARK_DIR`). C++ emulators are used where they exist: the reference
//! (`$LIBRISCV_CPP_EMULATOR`), the dispatch variants from
//! `scripts/build-cpp-reference.sh`, and `$LIBRISCV_CPP/emulator/rvlinux-asmjit`.
//!
//! Every run must print "Correct operation validated", and every variant
//! that counts instructions exactly must report the same count (asmjit's
//! counts are approximate and not compared). The guest's own timing is the
//! instruction counter (see `crates/testutil/guests/coremark`), so its
//! "Iterations/Sec" is not a speed; the score printed here is iterations per
//! wall-clock second.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use testutil::CppDispatch;

const MARCHES: &[(&str, &str)] = &[
    ("rv64imc_zicsr", "lp64"),
    ("rv64imafdc_zicsr", "lp64d"),
    ("rv64imafdc_zba_zbb_zbs_zicsr", "lp64d"),
];

struct Variant {
    name: String,
    emulator: PathBuf,
    flags: Vec<&'static str>,
    /// Whether its instruction count must match the others.
    exact: bool,
}

fn variants() -> Vec<Variant> {
    let rust = PathBuf::from(env!("CARGO_BIN_EXE_rvlinux"));
    let mut v = vec![Variant {
        name: "Rust interpreter".into(),
        emulator: rust.clone(),
        flags: vec!["--accurate"],
        exact: true,
    }];
    if cfg!(feature = "jit") {
        v.push(Variant {
            name: "Rust JIT (up front)".into(),
            emulator: rust.clone(),
            flags: vec!["--accurate", "--jit", "--no-background"],
            exact: true,
        });
        v.push(Variant {
            name: "Rust JIT (background)".into(),
            emulator: rust,
            flags: vec!["--accurate", "--jit"],
            exact: true,
        });
    }
    let interp = vec!["--no-translate", "--accurate"];
    let mut cpp_interp = |name: &str, path: PathBuf| {
        if path.exists() {
            v.push(Variant {
                name: name.into(),
                emulator: path,
                flags: interp.clone(),
                exact: true,
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
    let asmjit = testutil::cpp_source_dir()
        .join("emulator")
        .join("rvlinux-asmjit");
    if asmjit.exists() {
        v.push(Variant {
            name: "C++ asmjit (up front)".into(),
            emulator: asmjit.clone(),
            flags: vec!["--accurate", "--no-background"],
            exact: false,
        });
        v.push(Variant {
            name: "C++ asmjit (background)".into(),
            emulator: asmjit,
            flags: vec!["--accurate", "--background"],
            exact: false,
        });
    }
    v
}

fn env_num(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One run: wall time and the reported instruction count.
fn run(variant: &Variant, elf: &Path) -> (Duration, Option<u64>) {
    let start = Instant::now();
    let out = Command::new(&variant.emulator)
        .args(&variant.flags)
        .arg(elf)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", variant.emulator.display()));
    let elapsed = start.elapsed();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed = testutil::parse_run_output(&stdout, &String::from_utf8_lossy(&out.stderr));
    assert!(
        parsed.stdout.contains("Correct operation validated."),
        "{} failed on {}:\n{stdout}",
        variant.name,
        elf.display()
    );
    (elapsed, parsed.instructions)
}

fn main() {
    if !testutil::need_compiler() || !testutil::need_coremark() {
        return;
    }
    let iterations = env_num("COREMARK_ITERATIONS", 4000);
    let runs = env_num("COREMARK_RUNS", 3).max(1);
    let variants = variants();
    println!("CoreMark, {iterations} iterations, best of {runs} (wall-clock ms per process)\n");
    let mut header = format!("| {:<26} |", "");
    for (march, _) in MARCHES {
        header += &format!(" {:>28} |", march.trim_end_matches("_zicsr"));
    }
    println!("{header}");
    println!(
        "|{}|{}",
        "-".repeat(28),
        format!("{}:|", "-".repeat(29)).repeat(MARCHES.len())
    );

    let elfs: Vec<PathBuf> = MARCHES
        .iter()
        .map(|&(march, abi)| {
            testutil::build_coremark(march, abi, iterations).unwrap_or_else(|e| panic!("{e}"))
        })
        .collect();
    let mut counts: Vec<Option<u64>> = vec![None; elfs.len()];
    for variant in &variants {
        let mut line = format!("| {:<26} |", variant.name);
        for (col, elf) in elfs.iter().enumerate() {
            let mut best = Duration::MAX;
            for _ in 0..runs {
                let (t, count) = run(variant, elf);
                best = best.min(t);
                if variant.exact {
                    let expected = *counts[col].get_or_insert(count.unwrap_or(0));
                    assert_eq!(
                        count,
                        Some(expected),
                        "{}: instruction count differs on {}",
                        variant.name,
                        MARCHES[col].0
                    );
                }
            }
            let ms = best.as_secs_f64() * 1e3;
            let score = f64::from(iterations) / best.as_secs_f64();
            line += &format!(" {ms:>9.1} ms ({score:>7.0} it/s) |");
        }
        println!("{line}");
    }
    let mut line = format!("| {:<26} |", "Instructions");
    for count in &counts {
        line += &format!(" {:>28} |", count.map_or("-".into(), |c| c.to_string()));
    }
    println!("{line}");
}
