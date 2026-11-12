# libriscv-rs

A port of [libriscv](https://github.com/libriscv/libriscv) — a fast, sandboxed
RISC-V userspace emulator for embedding and scripting — to 100% safe Rust.

RV64 only.

**Status:** early. RV64IM freestanding guests run on the tail-call
interpreter and match the C++ emulator's output and instruction counts;
Linux userspace support is next. See [PORTING.md](PORTING.md) for scope,
design and progress.

## Building and testing

Requires the nightly toolchain pinned in `rust-toolchain.toml` (rustup picks
it up automatically). Instruction dispatch uses guaranteed tail calls
(`become`), which are not yet stable.

```sh
cargo build
cargo test
```

Most tests need a RISC-V cross compiler (`riscv64-linux-gnu-gcc-14`, override
with `RCC`) and the C++ reference emulator (`../libriscv/emulator/rvlinux`,
override with `LIBRISCV_CPP` or `LIBRISCV_CPP_EMULATOR`). Without them, those
tests are skipped; set `LIBRISCV_REQUIRE_TOOLCHAIN=1` to make them fail instead.

## License

BSD-3-Clause, same as libriscv.
