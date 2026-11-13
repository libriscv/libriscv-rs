# libriscv-rs

A port of [libriscv](https://github.com/libriscv/libriscv), a fast, sandboxed
RISC-V userspace emulator for embedding and scripting. The codebase uses
forbid unsafe, except for the optional Cranelift JIT. The port is modelled on
libriscv's architecture and internals, making it easy to compare the two
codebases for differences, with exception for the JIT backend.

Currently RV64GCB only.

## Benchmarks

| ELF | Rust interpreter | C++ threaded | C++ tailcall | Rust JIT | C++ asmjit |
|---|---:|---:|---:|---:|---:|
| rv64imafdc | 2425 | 2440 | 2197 | 19076 | 19326 |
| rv64imafdc_zba_zbb_zbs | 2777 | 2442 | 2290 | 19778 | 19640 |
| coremark-rv64g | 2492 | 2680 | 2380 | 18412 | 18836 |
| coremark-rv64gc | 2416 | 2330 | 2255 | 18700 | 19479 |
| coremark-rv64g_b | 2744 | 2716 | 2618 | 20839 | 20095 |
| coremark.rv64gb.gcc14.2.elf | 2809 | 2863 | 2596 | 17893 | 19131 |
| coremark-rv64g_zba_zbb | 2655 | 2779 | 2496 | 19559 | 19445 |

Note that this is with instruction counting and bounds-checked memory (~50% native with JIT).

## Building and testing

Requires the nightly toolchain pinned in `rust-toolchain.toml`. Instruction
dispatch uses guaranteed tail calls (`become`), which are not yet stable.

```sh
cargo build
cargo test
```

Most tests need a RISC-V cross compiler (`riscv64-linux-gnu-gcc-14`, override
with `RCC`) and the C++ reference emulator (`../libriscv/emulator/rvlinux`,
override with `LIBRISCV_CPP` or `LIBRISCV_CPP_EMULATOR`).

## JIT code cache

With the `jit` feature, the code Cranelift compiles can be saved and reused
by later runs: a cache of the JIT's output, not a separate compiler.

```sh
rvlinux --jit-save-shared prog.so prog.elf   # run once, save the compiled code
rvlinux --jit-load prog.so prog.elf          # later runs skip compiling
rvlinux -Q --jit-save-object prog.o prog.elf # or save it to link into rvlinux:
RVLINUX_JIT_EMBED=prog.o cargo build -p rvlinux --features jit
```

Saved code is only used with the same build, host CPU and memory layout it
was compiled for. Shared objects are linked with the system `cc`. In the
library, see `libriscv::jit_cache` and `MachineOptions::jit_cache_output`.

## Ahead-of-time translation to Rust

With the `aot` feature, a program's code can be translated to safe Rust
ahead of time (the counterpart of C++ libriscv's binary translation to C),
then compiled into the host program or into a shared object:

```sh
rvlinux -Q --aot-save prog.rs prog.elf          # translate to Rust source
RVLINUX_AOT_EMBED=prog.rs cargo build -p rvlinux --features aot   # compile it in: no unsafe anywhere
rvlinux -Q --aot-save-shared prog.so prog.elf   # or compile it with rustc into a shared object
rvlinux --aot-load prog.so prog.elf             # (feature aot-dylib)
```

Translated code has the interpreter's results and instruction counts, and
is only used for the segments and memory layout it was made for. Compiling
needs a nightly `rustc` (`become`), like the rest of the port. In the
library, see `libriscv::aot` and `MachineOptions::aot_output`.

## License

BSD-3-Clause, same as libriscv.
