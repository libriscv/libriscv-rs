//! Test helpers for the libriscv port.
//!
//! * [`build_and_load`] — cross-compile a C/C++ snippet to a RISC-V ELF
//!   (Rust equivalent of `tests/unit/codebuilder.cpp` in C++ libriscv).
//!   Outputs are cached under `target/riscv-guests/` keyed by a hash of the
//!   source, flags and compiler, so repeated test runs are fast.
//! * [`run_reference`] — run an ELF under the C++ `rvlinux` emulator and
//!   capture guest stdout + exit code, for differential testing.
//! * [`run_rust`] — the same, under this workspace's `rvlinux`.
//!
//! Environment variables:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `RCC` | `riscv64-linux-gnu-gcc-14` | C cross compiler |
//! | `RCXX` | `riscv64-linux-gnu-g++-14` | C++ cross compiler |
//! | `LIBRISCV_CPP` | `../libriscv` (sibling of the workspace) | C++ libriscv checkout |
//! | `LIBRISCV_CPP_EMULATOR` | `$LIBRISCV_CPP/emulator/rvlinux` | Reference emulator binary |
//! | `LIBRISCV_REQUIRE_TOOLCHAIN` | unset | If `1`, missing tools fail tests instead of skipping them |

#![forbid(unsafe_code)]

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

pub const DEFAULT_C_COMPILER: &str = "riscv64-linux-gnu-gcc-14";
pub const DEFAULT_CXX_COMPILER: &str = "riscv64-linux-gnu-g++-14";

/// Source language for [`build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    C,
    Cxx,
}

#[derive(Debug)]
pub enum BuildError {
    /// The cross compiler could not be executed at all.
    CompilerMissing(String),
    /// The compiler ran but rejected the code.
    CompileFailed {
        command: String,
        stderr: String,
    },
    Io(io::Error),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompilerMissing(cc) => write!(f, "RISC-V cross compiler not found: {cc}"),
            Self::CompileFailed { command, stderr } => {
                write!(f, "compilation failed: {command}\n{stderr}")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<io::Error> for BuildError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

fn env_or(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_owned())
}

/// Root of this Cargo workspace.
pub fn workspace_root() -> PathBuf {
    // crates/testutil -> workspace root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn target_dir() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("target"))
}

/// Path to the C++ libriscv checkout (`$LIBRISCV_CPP`, default `../libriscv`).
pub fn cpp_source_dir() -> PathBuf {
    std::env::var_os("LIBRISCV_CPP")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("..").join("libriscv"))
}

/// Path to the CoreMark sources (`$COREMARK_DIR`, default `../coremark`),
/// from <https://github.com/eembc/coremark>.
pub fn coremark_dir() -> PathBuf {
    std::env::var_os("COREMARK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("..").join("coremark"))
}

/// Skip-or-fail helper for tests that need the CoreMark sources.
pub fn need_coremark() -> bool {
    let dir = coremark_dir();
    tool_available(
        &format!("CoreMark sources in {} (set $COREMARK_DIR)", dir.display()),
        dir.join("core_main.c").exists(),
    )
}

/// Build freestanding CoreMark (performance run parameters, `iterations`
/// iterations) for `march` with our port (`guests/coremark`): output through
/// the `write` system call, time from `rdinstret` (so the output is
/// deterministic). ABIs with hardware floating point (`lp64d`) also print
/// the floating-point statistics.
pub fn build_coremark(march: &str, abi: &str, iterations: u32) -> Result<PathBuf, BuildError> {
    let dir = coremark_dir();
    let port = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("guests")
        .join("coremark");
    // The port is embedded so that changing it invalidates the cached ELF
    // (core_portme.h is found through -I; it is part of the key as text).
    let mut code = String::from(include_str!("../guests/coremark/core_portme.c"));
    code.push_str("\n/* core_portme.h:\n");
    code.push_str(&include_str!("../guests/coremark/core_portme.h").replace("*/", "* /"));
    code.push_str("*/\n");
    let sources = [
        "core_list_join.c",
        "core_main.c",
        "core_matrix.c",
        "core_state.c",
        "core_util.c",
    ]
    .map(|file| dir.join(file));
    let has_float = u8::from(abi.ends_with('d') || abi.ends_with('f'));
    let args = format!(
        "-march={march} -mabi={abi} -O2 -mno-relax -static -nostdlib -ffreestanding \
         -I{} -I{} -DPERFORMANCE_RUN=1 -DITERATIONS={iterations} -DHAS_FLOAT={has_float}",
        port.display(),
        dir.display()
    );
    build_with_sources(&code, &sources, &args, Lang::C)
}

/// CoreMark 1.0 as a regular Linux executable, for benchmarking: the POSIX
/// port of `$COREMARK_DIR` (`posix/core_portme.c`) linked statically
/// against glibc, `-O3`, performance run parameters and `ITERATIONS=0`, so
/// CoreMark calibrates itself to at least 10 seconds of real time (from
/// `clock_gettime`/`gettimeofday` system calls) and its "Iterations/Sec"
/// is the score. The output is not deterministic: for difftests use
/// [`build_coremark_linux`].
pub fn build_coremark_timed(march: &str, abi: &str) -> Result<PathBuf, BuildError> {
    let dir = coremark_dir();
    // The port's text is part of the cache key, so that editing it (a
    // checkout may carry local changes) rebuilds the ELF.
    let mut code = String::from("/* CoreMark 1.0, posix port:\n");
    for file in ["posix/core_portme.c", "posix/core_portme.h"] {
        let text = std::fs::read_to_string(dir.join(file))?;
        code.push_str(&text.replace("*/", "* /"));
    }
    code.push_str("*/\n");
    let sources = [
        "core_list_join.c",
        "core_main.c",
        "core_matrix.c",
        "core_state.c",
        "core_util.c",
        "posix/core_portme.c",
    ]
    .map(|file| dir.join(file));
    let flags = format!("-march={march} -mabi={abi} -O3 -static");
    let args = format!(
        "{flags} -I{} -I{} -DPERFORMANCE_RUN=1 -DITERATIONS=0 -DFLAGS_STR=\"{}\"",
        dir.join("posix").display(),
        dir.display(),
        flags.replace(' ', ","),
    );
    build_with_sources(&code, &sources, &args, Lang::C)
}

/// Build CoreMark as a regular Linux executable for difftests: the POSIX
/// port (`posix/core_portme.c`) linked statically against glibc,
/// performance run parameters, `iterations` iterations. Time comes from
/// `clock()`, which is redirected to `rdinstret` so that the output (and
/// the instruction count) is deterministic and can be compared with the
/// C++ emulator; its "Iterations/Sec" is therefore not a speed. To
/// benchmark, use [`build_coremark_timed`].
pub fn build_coremark_linux(
    march: &str,
    abi: &str,
    iterations: u32,
) -> Result<PathBuf, BuildError> {
    let dir = coremark_dir();
    let code = "#include <time.h>\n\
        /* clock() for CoreMark: 4 ticks per retired instruction, at\n\
           CLOCKS_PER_SEC (1 MHz): a nominal 250k instructions per\n\
           second, so that short runs pass the \"at least 10 secs\"\n\
           rule (10 iterations are ~14 seconds). */\n\
        clock_t coremark_instret_clock(void) {\n\
            unsigned long v;\n\
            __asm__ volatile(\"rdinstret %0\" : \"=r\"(v));\n\
            return (clock_t)(v * 4);\n\
        }\n";
    let sources = [
        "core_list_join.c",
        "core_main.c",
        "core_matrix.c",
        "core_state.c",
        "core_util.c",
        "posix/core_portme.c",
    ]
    .map(|file| dir.join(file));
    let flags = format!("-march={march} -mabi={abi} -O2 -static");
    let args = format!(
        "{flags} -I{} -I{} -DPERFORMANCE_RUN=1 -DITERATIONS={iterations} \
         -DSEED_METHOD=SEED_VOLATILE -DUSE_CLOCK=1 -Dclock=coremark_instret_clock \
         -DFLAGS_STR=\"{}\"",
        dir.join("posix").display(),
        dir.display(),
        flags.replace(' ', ","),
    );
    build_with_sources(code, &sources, &args, Lang::C)
}

/// Prebuilt guest ELFs shipped with C++ libriscv (`tests/unit/elf`).
pub fn cpp_fixture(name: &str) -> PathBuf {
    cpp_source_dir()
        .join("tests")
        .join("unit")
        .join("elf")
        .join(name)
}

fn require_toolchain() -> bool {
    std::env::var("LIBRISCV_REQUIRE_TOOLCHAIN").is_ok_and(|v| v == "1")
}

/// Call at the top of a test that needs an external tool. Returns `false`
/// (and prints why) if the test should be skipped; panics instead when
/// `LIBRISCV_REQUIRE_TOOLCHAIN=1` so CI never silently skips.
pub fn tool_available(what: &str, ok: bool) -> bool {
    if !ok {
        assert!(
            !require_toolchain(),
            "{what} is required (LIBRISCV_REQUIRE_TOOLCHAIN=1) but not available"
        );
        eprintln!("SKIPPED: {what} not available");
    }
    ok
}

fn runs(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The configured C or C++ cross compiler.
pub fn compiler(lang: Lang) -> String {
    match lang {
        Lang::C => env_or("RCC", DEFAULT_C_COMPILER),
        Lang::Cxx => env_or("RCXX", DEFAULT_CXX_COMPILER),
    }
}

/// True if the cross compiler for `lang` can be executed.
pub fn have_compiler(lang: Lang) -> bool {
    runs(&compiler(lang))
}

/// Skip-or-fail helper: `if !testutil::need_compiler() { return; }`
pub fn need_compiler() -> bool {
    tool_available(
        &format!("cross compiler `{}`", compiler(Lang::C)),
        have_compiler(Lang::C),
    )
}

/// Compile `code` with extra compiler `args` (whitespace-separated) and
/// return the path to the cached ELF.
pub fn build(code: &str, args: &str, lang: Lang) -> Result<PathBuf, BuildError> {
    build_with_sources(code, &[], args, lang)
}

/// [`build`] with further source files compiled and linked along with
/// `code`. The cache key covers their paths, not their contents.
pub fn build_with_sources(
    code: &str,
    sources: &[PathBuf],
    args: &str,
    lang: Lang,
) -> Result<PathBuf, BuildError> {
    let cc = compiler(lang);
    let mut h = DefaultHasher::new();
    (code, sources, args, lang, &cc).hash(&mut h);
    let key = format!("{:016x}", h.finish());

    let dir = target_dir().join("riscv-guests");
    std::fs::create_dir_all(&dir)?;
    let elf = dir.join(format!("{key}.elf"));
    if elf.exists() {
        return Ok(elf);
    }
    let (ext, std) = match lang {
        Lang::C => ("c", "-std=gnu11"),
        Lang::Cxx => ("cpp", "-std=gnu++17"),
    };
    // Unique names per call: test threads in one process may build the same
    // guest concurrently, so the source and output can't be shared. The ELF
    // is renamed into place, so nobody sees a half-written one.
    static BUILDS: AtomicU64 = AtomicU64::new(0);
    let unique = format!(
        "{}.{}",
        std::process::id(),
        BUILDS.fetch_add(1, Ordering::Relaxed)
    );
    let src = dir.join(format!("{key}.{unique}.{ext}"));
    std::fs::write(&src, code)?;
    let tmp = dir.join(format!("{key}.{unique}.tmp"));

    let mut cmd = Command::new(&cc);
    cmd.arg(std)
        .arg("-o")
        .arg(&tmp)
        .args(args.split_whitespace())
        .arg(&src)
        .args(sources)
        .arg("-lm");
    let command = format!("{cmd:?}");
    let out = cmd
        .output()
        .map_err(|_| BuildError::CompilerMissing(cc.clone()))?;
    if !out.status.success() {
        return Err(BuildError::CompileFailed {
            command,
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    std::fs::rename(&tmp, &elf)?;
    let _ = std::fs::remove_file(&src);
    Ok(elf)
}

/// Compile a C snippet and return the ELF bytes. Typical args: `"-O2 -static"`.
pub fn build_and_load(code: &str, args: &str) -> Result<Vec<u8>, BuildError> {
    Ok(std::fs::read(build(code, args, Lang::C)?)?)
}

/// Like [`build_and_load`] but for C++.
pub fn build_and_load_cxx(code: &str, args: &str) -> Result<Vec<u8>, BuildError> {
    Ok(std::fs::read(build(code, args, Lang::Cxx)?)?)
}

/// Result of running a guest program under an emulator binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutput {
    /// Everything the guest (and emulator) wrote to stdout, *before* the
    /// `>>> Program exited` summary line.
    pub stdout: String,
    /// Parsed from `>>> Program exited, exit code = N`. `None` if absent
    /// (e.g. the emulator crashed or was run with `--silent`).
    pub exit_code: Option<i64>,
    /// Parsed from `Instructions executed: N` when run with `--accurate`.
    pub instructions: Option<u64>,
    /// Raw stderr.
    pub stderr: String,
}

const EXIT_MARKER: &str = ">>> Program exited, exit code = ";
const INSN_MARKER: &str = "Instructions executed: ";

/// Split emulator output into guest stdout and the trailing summary.
/// Both `rvlinux` implementations must print the summary in this format.
pub fn parse_run_output(stdout: &str, stderr: &str) -> RunOutput {
    let (guest, summary) = match stdout.rfind(EXIT_MARKER) {
        Some(i) => (&stdout[..i], &stdout[i..]),
        None => (stdout, ""),
    };
    let exit_code = summary
        .strip_prefix(EXIT_MARKER)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|n| n.parse().ok());
    let instructions = summary
        .find(INSN_MARKER)
        .and_then(|i| summary[i + INSN_MARKER.len()..].split_whitespace().next())
        .and_then(|n| n.parse().ok());
    RunOutput {
        stdout: guest.to_owned(),
        exit_code,
        instructions,
        stderr: stderr.to_owned(),
    }
}

/// Path to the C++ reference `rvlinux`.
pub fn reference_emulator() -> PathBuf {
    std::env::var_os("LIBRISCV_CPP_EMULATOR")
        .map(PathBuf::from)
        .unwrap_or_else(|| cpp_source_dir().join("emulator").join("rvlinux"))
}

/// C++ interpreter dispatch modes built by `scripts/build-cpp-reference.sh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppDispatch {
    /// Computed goto (C++ default).
    Threaded,
    /// clang `musttail` + `preserve_none` — the design the Rust port mirrors.
    Tailcall,
    /// Plain `switch`.
    Switch,
}

impl CppDispatch {
    pub const ALL: [Self; 3] = [Self::Threaded, Self::Tailcall, Self::Switch];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Threaded => "threaded",
            Self::Tailcall => "tailcall",
            Self::Switch => "switch",
        }
    }
}

/// Path to a C++ `rvlinux` built in a specific dispatch mode by
/// `scripts/build-cpp-reference.sh` (`target/cpp-ref/<mode>/rvlinux`).
pub fn reference_emulator_variant(mode: CppDispatch) -> PathBuf {
    target_dir()
        .join("cpp-ref")
        .join(mode.name())
        .join("rvlinux")
}

/// Path to the C++ `rvlinux` with asmjit built by
/// `scripts/build-cpp-reference.sh asmjit` (`target/cpp-ref/asmjit/rvlinux`).
pub fn reference_asmjit_emulator() -> PathBuf {
    target_dir().join("cpp-ref").join("asmjit").join("rvlinux")
}

/// Skip-or-fail helper for tests that diff against the C++ emulator.
pub fn need_reference() -> bool {
    let emu = reference_emulator();
    tool_available(
        &format!("reference emulator {}", emu.display()),
        emu.exists(),
    )
}

fn run_emulator(emu: &Path, flags: &[&str], elf: &Path, args: &[&str]) -> io::Result<RunOutput> {
    let out = Command::new(emu)
        .args(flags)
        .arg(elf)
        .args(args)
        .stdin(Stdio::null())
        .output()?;
    Ok(parse_run_output(
        &String::from_utf8_lossy(&out.stdout),
        &String::from_utf8_lossy(&out.stderr),
    ))
}

/// Run `elf` under the C++ reference emulator in pure-interpreter mode
/// (`--no-translate`) with exact instruction counting (`--accurate`).
pub fn run_reference(elf: &Path, args: &[&str]) -> io::Result<RunOutput> {
    run_reference_with(elf, &[], args)
}

/// [`run_reference`] with extra emulator flags (e.g. `["--fuel", "1000"]`).
pub fn run_reference_with(elf: &Path, flags: &[&str], args: &[&str]) -> io::Result<RunOutput> {
    let mut all = vec!["--no-translate", "--accurate"];
    all.extend_from_slice(flags);
    run_emulator(&reference_emulator(), &all, elf, args)
}

/// Run `elf` under a Rust `rvlinux` binary with the same flags as
/// [`run_reference`]. Pass `env!("CARGO_BIN_EXE_rvlinux")` from tests in
/// the `rvlinux` crate.
pub fn run_rust(rvlinux: &Path, elf: &Path, args: &[&str]) -> io::Result<RunOutput> {
    run_rust_with(rvlinux, elf, &[], args)
}

/// [`run_rust`] with extra emulator flags.
pub fn run_rust_with(
    rvlinux: &Path,
    elf: &Path,
    flags: &[&str],
    args: &[&str],
) -> io::Result<RunOutput> {
    let mut all = vec!["--accurate"];
    all.extend_from_slice(flags);
    run_emulator(rvlinux, &all, elf, args)
}

/// The `>>> Machine exception ...` line in emulator output, if any. Both
/// emulators print it identically; C++ prints more context around it.
pub fn exception_line(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        .find(|l| l.starts_with(">>> Machine exception"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_summary() {
        let out = parse_run_output(
            "Hello\n>>> Program exited, exit code = -1 (0xFFFFFFFFFFFFFFFF)\n\
             Instructions executed: 12345  Runtime: 0.1ms  Insn/s: 100mi/s\n",
            "",
        );
        assert_eq!(out.stdout, "Hello\n");
        assert_eq!(out.exit_code, Some(-1));
        assert_eq!(out.instructions, Some(12345));
    }

    #[test]
    fn parses_missing_summary() {
        let out = parse_run_output(">>> Machine exception 2: x\n", "");
        assert_eq!(out.exit_code, None);
        assert_eq!(out.instructions, None);
    }
}
