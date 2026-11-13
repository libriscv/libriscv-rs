# libriscv-rs

Safe-Rust port of C++ libriscv (a RISC-V userspace emulator for sandboxing and
scripting). **Read `PORTING.md` first**: scope, architecture decisions, phase
plan and the C++ → Rust file mapping live there.

## Where things are

- C++ reference source: `../libriscv` (or `$LIBRISCV_CPP`). Library code is in
  `../libriscv/lib/libriscv/`; `../libriscv/AGENTS.md` is its architecture guide.
- C++ reference emulator (the test oracle): `../libriscv/emulator/rvlinux`.
- CoreMark sources (difftest and benchmark): `../coremark` (or `$COREMARK_DIR`).
- `crates/libriscv` — the library. `crates/rvlinux` — CLI. `crates/testutil` —
  guest cross-compilation and oracle helpers (test-only).

## Commands

```sh
cargo test                                   # all tests (tool-dependent tests skip if tools missing)
LIBRISCV_REQUIRE_TOOLCHAIN=1 cargo test      # same, but fail instead of skipping
cargo test -- --ignored                      # include not-yet-enabled difftests
cargo clippy --all-targets -- -D warnings    # must be clean (also with --features libriscv/jit,rvlinux/jit)
cargo test --workspace --features libriscv/jit,rvlinux/jit   # + Cranelift JIT, difftests also run --jit
cargo test --workspace --features libriscv/jit,rvlinux/jit,libriscv/aot-dylib,rvlinux/aot-dylib  # + AOT (needs rustc)
rvlinux --jit-save-shared x.so prog.elf; rvlinux --jit-load x.so prog.elf   # JIT code cache (also --jit-save-object, RVLINUX_JIT_EMBED)
rvlinux --aot-save-shared x.so prog.elf; rvlinux --aot-load x.so prog.elf   # AOT: translate to safe Rust (also --aot-save x.rs, RVLINUX_AOT_EMBED)
```

Expected durations (warm build, 32 cores): the whole suite ~25 s with the JIT
features (`difftest.rs` ~14 s, bounded by one up-front JIT compile of static
libstdc++), ~7 s without. Cranelift is always built optimized
(`[profile.dev.package."*"]`); our crates stay at opt-level 0. A test that
loops over many runs should be split into separate tests (see `difftests!`)
so the harness runs them concurrently. Never `git stash` work to time HEAD.

```sh
cargo fmt --all
scripts/difftest.sh prog.elf [args...]       # diff one ELF against the C++ emulator
scripts/build-cpp-reference.sh               # C++ threaded/tailcall/switch builds -> target/cpp-ref/
cargo bench -p rvlinux --features jit --bench coremark   # CoreMark: Rust interpreter/JIT vs C++
fuzz/do_fuzz.sh vmfuzzer -- -max_total_time=300   # libFuzzer (vmfuzzer/elffuzzer/syshelperfuzzer); needs cargo-fuzz
```

Toolchain: pinned nightly (`rust-toolchain.toml`) for `become`. Don't bump it
as a side effect of other work; see "Toolchain" in `PORTING.md`.

Guests compile with `riscv64-linux-gnu-gcc-14` (`$RCC`); outputs are cached in
`target/riscv-guests/`.

## Rules

- **No `unsafe`.** Enforced by `forbid(unsafe_code)`; never work around it
  (no `allow`, no unsafe-using helper crates added to dodge it). The one
  exception is `crates/libriscv-jit-rt`, the executable-memory runtime of the
  optional `jit` feature (it also loads AOT shared objects for `aot-dylib`):
  keep its unsafe surface minimal and documented, and keep everything else
  (including `libriscv/src/jit/` and `aot/`) unsafe-free. Code the AOT
  translator generates is safe Rust too.
- **Guest input must never panic the host.** Return `MachineError`. Use
  `get`/`get_mut`, checked/wrapping arithmetic and `try_into` on anything a
  guest controls. The clippy lints `indexing_slicing`, `unwrap_used`,
  `expect_used`, `panic` warn in non-test code; fix them rather than allowing
  them, except where an invariant is provable locally (then `#[allow]` with a
  comment stating the invariant).
- **Match the C++ interpreter exactly** — results, exception kinds, syscall
  behaviour and retired-instruction counts. When unsure what's correct, the C++
  source and the oracle decide, not the spec alone. If you find a C++ bug, note
  it in the PR and keep parity unless told otherwise.
- **Port the tests with the code.** Each ported C++ file should bring the
  corresponding `../libriscv/tests/unit/*.cpp` cases along as Rust tests.
- **Dispatch tail calls must use `become`**, never a plain call in tail
  position. Handler chains are unbounded, and only `become` guarantees they
  don't grow the stack. All handlers share one signature (see "Dispatch" in
  `PORTING.md`). For dispatch structure, `../libriscv/lib/libriscv/tailcall_dispatch.cpp`
  is the reference.
- Start each new module with a `//! Port of <C++ files>.` header.
- Keep `rvlinux`'s summary output byte-compatible with the C++ emulator
  (`>>> Program exited, exit code = N (0xN)`, `Instructions executed: N ...`);
  `testutil::parse_run_output` depends on it.
- Update the checkboxes/status in `PORTING.md` in the same commit as the work.
- No new runtime dependencies in `libriscv` without asking. Dev-dependencies
  are fine.
- Out-of-scope C++ components (binary translation to C, libtcc, asmjit, asm
  dispatch, win32, RV32, RV128, C API) are listed in `PORTING.md`; don't port
  them. The JIT is Cranelift-based instead, and AOT translates to safe Rust
  (see "JIT" and "AOT" in `PORTING.md`). The port
  is RV64 only: no width generics, no `W == 4` paths.
- Where C++'s tailcall and threaded dispatchers disagree, follow the threaded
  one (`cpu_dispatch.cpp`): it's what the oracle runs. See "Dispatch" in
  `PORTING.md`.

## Integration tests

Integration test files start with
`#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]`
because clippy's in-test exemptions don't cover helper functions. Gate tests
needing external tools with `if !testutil::need_compiler() { return; }` /
`testutil::need_reference()`.
