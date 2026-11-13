# libriscv-rs fuzzing

[cargo-fuzz] harnesses built on libFuzzer: a port of libriscv's `fuzz/`
(`fuzz.cpp`, the first four targets), plus targets for the parts of the
library C++ doesn't fuzz. The harnesses are in `src/` (`lib.rs` for the
C++ ones), and each target in `fuzz_targets/` is a one-line wrapper around
one of them.

| target | C++ mode | what it fuzzes |
| ------ | -------- | -------------- |
| `vmfuzzer` | `FUZZ_VM` | the decoder and interpreter: the input is code at 0x2000, run for 5000 instructions |
| `elffuzzer` | `FUZZ_ELF` | the ELF loader, execute segments and the Linux system calls (no filesystem, no sockets): the input is a program, run for 5000 instructions |
| `nativefuzzer` | `FUZZ_NAT` | the native heap and memory system calls (`setup_native_heap`, `setup_native_memory`): the input is a system call number and code at 0x2000 that can call them, run for 10000 instructions, then that system call once more |
| `syshelperfuzzer` | `FUZZ_SYSH` | the memory helpers host system calls are built from (`memstring`, `gather_buffers`, `memview`, `memcmp`, `memcpy`, `gather_writable_buffers`), called from system call 1 |
| `linuxfuzzer` | | the Linux system calls, POSIX threads, signal delivery after a fault, and the sandboxed file system (read-only `/dev/null`, `/dev/zero`, `/dev/urandom`, `/etc/hosts`; no sockets, no proxy mode): the input is a program (wrapped in a minimal ELF, with argv/envp/auxv), then one more system call |
| `libcfuzzer` | | the libc fast path (`libc_fastpath`): the input places the patched functions in the program by symbol and calls them |
| `forkfuzzer` | | forks and copy-on-write: run, fork, then run both; a fork must end exactly like its parent, and a fork of the fork must be unaffected by both |
| `guestfuzzer` | | the guest C++ containers (`std::string`, `vector`, `unordered_map`, `variant`): host operations on objects whose bytes the input supplies |
| `vmcallfuzzer` | | VM calls with host arguments, nested preemption from system calls, dyncalls, and the system call argument readers |
| `jitdifffuzzer` | | the JIT against the interpreter (`--features jit`): registers, PC, counter, error and memory must match exactly. Inputs with NaN-producing FP arithmetic are skipped (NaN bits are the host's in the JIT). Compares only on x86-64, the JIT's only host so far |

The ELF-based targets put the code at 0x10000 and a zero-filled read-write
segment at 0x200000, with some argument registers pointing into it.
`dicts/<target>.dict` gives libFuzzer the instruction words, system call
numbers, sandbox paths and guest pointers those targets need;
`do_fuzz.sh` passes it. `cargo test` checks that every harness reaches what
it fuzzes (a harness whose setup fails fuzzes nothing, silently).

`nativefuzzer` runs without the native thread system calls, which aren't
ported. There are no 32- or 128-bit variants: the port is RV64 only.

Guest input must never panic the host, so every crash is a bug. Errors from
the machine are expected and ignored, like the C++ harness ignores
exceptions.

## Setup

```sh
cargo install cargo-fuzz
```

The crate isn't part of the main workspace (`cargo test` shouldn't build
libFuzzer), but it uses the repository's pinned nightly.

## Running

```sh
./do_fuzz.sh vmfuzzer                            # until a crash or Ctrl-C
./do_fuzz.sh vmfuzzer -- -max_total_time=300     # 5 minutes
./do_fuzz.sh elffuzzer -- -fork=16               # 16 worker processes
cargo fuzz run -s none -a syshelperfuzzer        # without the script
```

`do_fuzz.sh` runs `cargo fuzz run -s none -a` (override with `FUZZ_FLAGS`)
and forwards arguments after `--` to libFuzzer:

- `-s none`: no sanitizer. The library is safe Rust, so AddressSanitizer
  finds nothing a bounds check doesn't already turn into a panic, and costs
  throughput. With the `jit` feature (which brings
  `libriscv-jit-rt`'s unsafe code along) use `FUZZ_FLAGS="-s address -a
  --features jit"` instead.
- `-a`: debug assertions and overflow checks. An arithmetic overflow on
  guest-controlled values is a bug even where release builds would wrap.

The corpus is kept in `corpus/<target>/`, crashing inputs in
`artifacts/<target>/` (both git-ignored). Replay one with:

```sh
cargo fuzz run -s none -a vmfuzzer artifacts/vmfuzzer/crash-<hash>
```

then turn it into a regression test next to the code it found a bug in.

`-fork=N` is the easiest way to use more cores. In fork mode libFuzzer
stops at the first crash, like a single process does.

The decoder without the C extension (its 4-byte layout):

```sh
FUZZ_FLAGS="-s none -a --no-default-features --target-dir target/no-c" ./do_fuzz.sh vmfuzzer
```

### Long runs

`overnight.sh` fuzzes every target at once for a given time (default 8
hours), splitting the cores between them, and doesn't stop at findings:
crashes, OOMs (over 2 GB) and timeouts (over 30 s) are written to
`artifacts/<target>/` and fuzzing continues. Logs go to `logs/`, the corpora
grow in `corpus/` across runs. It ends with a table of findings per target.

```sh
./overnight.sh                          # all targets, 8 hours, all cores
JOBS=16 ./overnight.sh 3600 guestfuzzer # one target, 16 workers, an hour
```

`jitdifffuzzer` is built in `target/jit`: fork mode re-runs the target
binary for each job, so rebuilding the shared `target/` with the JIT would
switch the other targets to it. Set `NO_JIT=1` to skip it where the JIT
doesn't run (anything but x86-64).

### Coverage

```sh
rustup component add llvm-tools-preview   # once, for the pinned nightly
cargo fuzz coverage -s none -a linuxfuzzer corpus/linuxfuzzer
```

writes `coverage/linuxfuzzer/coverage.profdata`; report it with the
toolchain's `llvm-cov report` against
`target/<triple>/coverage/<triple>/release/linuxfuzzer`.

### If the fuzzer prints nothing and appears to hang

libFuzzer runs `llvm-symbolizer` for `NEW_FUNC` lines and stack traces and
waits for it. On distros with a default debuginfod URL (Ubuntu sets
`DEBUGINFOD_URLS` from `/etc/profile.d`) its debuginfod client can hang
forever. `do_fuzz.sh` clears `DEBUGINFOD_URLS`; do the same when running
`cargo fuzz` directly.

## Seeding the ELF loader fuzzer

Starting from an empty corpus, `elffuzzer` spends the whole run failing ELF
header validation. `do_fuzz.sh elffuzzer` seeds it with the 64-bit RISC-V
binaries in `$LIBRISCV_CPP/tests/unit/elf` (default `../../libriscv`),
copied into `seeds/elffuzzer/` and passed after the writable
`corpus/elffuzzer/`, so libFuzzer never writes into the seeds. Don't set
`-max_len`: libFuzzer skips seed files larger than it, which drops most of
these binaries. The large seeds trade throughput for coverage; see the C++
`fuzz/README.md` for measurements.

## Fuzzing in CI

`.github/workflows/fuzzing.yml` lints and builds the fuzz crate, then runs
each target (and `vmfuzzer` without the C extension) for 30 seconds on
push and pull request (from an empty corpus,
`elffuzzer` seeded as above), failing the job on a crash and uploading the
reproducer as an artifact. Use the workflow's *Run workflow* button to fuzz
longer. It catches regressions; it is no substitute for long local runs.

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
