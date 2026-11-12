# Porting libriscv to safe Rust

This is the plan and progress tracker for the port. Update the status column
in the same commit that changes it.

**Reference:** C++ libriscv at `../libriscv` (override with `$LIBRISCV_CPP`),
tracking upstream commit `6b81381257f00826da357c40e7288a0e5e523274` (v1.11+).
When the C++ side moves, bump this hash and the one in `.github/workflows/ci.yml`
together.

## Goals

1. **Safe:** `#![forbid(unsafe_code)]` in every crate. No exceptions inside this
   workspace. Prefer zero runtime dependencies for `libriscv`. Nightly
   *language features* are fine (see Toolchain); unsafe is not.
2. **Host never panics on guest input.** Every guest fault is a `MachineError`.
   `clippy::{indexing_slicing, unwrap_used, expect_used, panic}` warn in
   non-test code to keep this honest.
3. **Behaviourally identical to the C++ interpreter.** Same guest-visible
   semantics, same syscall results, same exception kinds, and the same
   retired-instruction count in `--accurate` mode. The C++ `rvlinux` is the
   oracle (see Testing).
4. **Fast.** Match C++ tailcall dispatch and get close to C++ threaded
   (computed goto), which is ~4% faster on coremark (see Baseline). Measured,
   not guessed: coremark, dhrystone and STREAM from `../libriscv/binaries`.
5. **Ergonomic embedding API**, idiomatic Rust rather than a transliteration of
   the C++ templates.

## Scope

In scope: the interpreter, **RV64 only**, extensions I M A C F D Zba Zbb Zbc
Zbs Zicond Zbkb Zfh(min) Zcb Zimop/Zcmop and V; ELF loading; flat arena memory and
(later) virtual paging; Linux, newlib and minimal syscall sets; vmcall/preempt;
fork/CoW; serialization; threads; native heap/libc; debugger, disassembler and
GDB stub.

**Out of scope** (inherently need `unsafe` or native code generation):

| C++ component | Files | Why |
|---|---|---|
| Binary translation | `tr_*.cpp`, `tr_*.hpp` | Compiles C and `dlopen`s it |
| libtcc JIT | `tr_tcc.cpp` | Same |
| asmjit JIT | `asmjit/` | Emits and executes machine code |
| Assembly dispatch | `amd64/` | Hand-written asm |
| Windows-specific shims | `win32/` | Revisit after Linux parity; most of it is replaced by `std` |
| RV32 | `rv32i.cpp`, the `W == 4` paths everywhere | Not needed; dropped on 2026-11-12 to keep the port non-generic |
| RV128 | `rv128i*.cpp`, `rva128_instr.cpp` | Low value |
| C API | `../libriscv/c/` | An `extern "C"` API needs `unsafe`; if wanted, put it in a separate crate outside the forbid policy |

## Architecture decisions

Decided (implemented in Phase 0):

- **RV64 only, no width generics.** C++ `template <int W>` is gone: registers
  and guest addresses are `u64`, and `Machine`, `Cpu` and `Memory` are plain
  types. (Phase 0 had a `Width` trait for RV32/RV64; it was removed when RV32
  went out of scope.)
- **Errors, not exceptions.** `MachineException` → `MachineError { kind, message, data }`
  with `ExceptionKind` mirroring the C++ enum in order. The dispatch loop
  returns `Result`; there is no unwinding (`panic = "abort"` in release).
- **Instruction formats.** `union rv32i_instruction` → `Instr(u32)` with
  zero-cost per-format views (`instr.itype().signed_imm()`).
- **Dispatch: guaranteed tail calls.** See "Dispatch" below.
- **CMake options → Cargo features** where they change behaviour
  (`ext-a`, `ext-c`, `ext-f`, `ext-v`, `fcsr`, `debug-checks`). Options that
  only select a dispatch strategy or JIT have no equivalent. `ext-c` also
  selects the decoder layout (one entry per 2 bytes, as C++
  `compressed_enabled`), so it changes block structure and therefore where
  timeouts land; the default build matches the C++ default.

Decided in Phase 1:

- **Memory representation.** The C++ flat read-write arena without paging
  (`Vec<u8>`, zero-allocated so the OS commits it lazily, plus one page of
  padding so a wide access at the last address needs no size check).
  Protection is the two compares of `memory_inline.hpp`: reads in
  `[0x1000, arena_size)`, writes in `[initial_rodata_end, arena_size)`. There
  is no page attribute table: C++ without `RISCV_VIRTUAL_PAGING` has none
  either (`set_page_attr` is a no-op). Virtual paging comes later as a second
  backend behind an enum, not as a type parameter on everything.
- **Machine shape.** `Machine { cpu, memory, .. }` is not generic. System
  call handlers are `fn(&mut Machine) -> Result<()>` in a per-machine table
  (C++'s is per process). Userdata is still open (Phase 3).
- **One definition per instruction.** Each instruction is a unit type
  implementing `isa::Op` (the C++ precise handler). The dispatcher wraps the
  same `exec` in its generic handlers (`seq::<O>`, `function::<O>`,
  `funcblock::<O>`), so the fast and precise interpreters can't drift
  apart; only block-ending instructions have dedicated tail-call handlers.

To decide at the start of the phase that needs them (recommendation given):
- **Host access to guest memory (Phase 3).** C++ hands out raw pointers. Rust
  hands out `&[u8]` / `&mut [u8]` borrowing the memory, plus typed helpers
  (`read::<T>`, `write`, `memstring`, `memview`). Zero-copy is preserved; the
  borrow checker replaces the "pointer is valid until the next mmap" rule.
  POD reinterpretation goes through explicit `from_le_bytes` (no `bytemuck`
  unless we accept the dependency).
- **Syscall handlers and userdata (Phase 3).** Recommend handlers as plain
  `fn(&mut Machine<W, U>) -> Result<()>` stored in a fixed table (no boxing,
  matches C++ `syscall_t*`), and userdata as a generic `Machine<W, U = ()>`
  field so handlers can borrow `machine.memory` and `machine.userdata`
  disjointly. Avoid `Box<dyn Any>` downcasts on the hot path.
- **Floating point.** Use `f32`/`f64` with explicit NaN-boxing helpers
  (`rvfd.hpp`). Rounding modes/flags only under the `fcsr` feature, like C++.
- **Threads/fork.** No global state; `Machine` should be `Send`. Fork shares
  read-only pages via `Arc`.

## Toolchain

Pinned nightly in `rust-toolchain.toml` (currently `nightly-2026-09-28`,
rustc `d080e7dff`). We need it for `become` (`feature(explicit_tail_calls)`).
The feature is still flagged *incomplete* upstream, so a miscompile is
possible; the difftests and instruction-count checks are the safety net.

Bumping the pin is a deliberate change of its own: update `rust-toolchain.toml`,
run the full test suite including `--ignored`, re-run the benchmarks and record
any change in Baseline. If `become` stabilises, move to stable and drop the
`#![feature]`. `tail_call_tests.rs` checks that indirect `become` is still
a real tail call in an unoptimised build (it overflows the stack otherwise).

## Dispatch

Decision: **tail-call threaded dispatch with `become`**, modelled on C++
`tailcall_dispatch.cpp` (clang `musttail` + `preserve_none`). It's the fastest
dispatch available in safe Rust. Each handler ends by tail-calling the next
handler through a function pointer. So every instruction has its own indirect
branch, which is what makes computed goto fast. `match` gives one shared branch,
and on the C++ side `switch` is ~28% slower than computed goto on coremark.

How the C++ version maps over:

| C++ (`tailcall_dispatch.cpp`) | Rust |
|---|---|
| `INSTRUCTION(bc, name)`: `static TcoRet<W> name(DecoderData<W>* d, DecodedExecuteSegment<W>* exec, CPU<W>& cpu, addr_t pc, InstrCounter& counter)` | `fn name(exec: &DecodedSegment<W>, i: usize, cpu: &mut Cpu<W>, pc: W::Addr, counter: Counter) -> Exit` with one signature for all handlers |
| `MUSTTAIL return computed_opcode<W>[bc](...)` | `become (entry.handler)(...)`. Store the handler `fn` in the decoder entry itself, so there's no second table lookup |
| `d += 1` pointer bump | index `i + 1`; read with `exec.cache.get(i)`. That's one bounds check per instruction: measure it before trying to remove it |
| `__attribute__((preserve_none))` | `extern "rust-preserve-none" fn` (`feature(rust_preserve_none_cc)`, works on the pinned nightly). Try it in Phase 4 and keep it if benchmarks say so |
| `BEGIN_BLOCK()` adds the block's instruction count; `OVERFLOW_CHECK()` only on branches/jumps | Same. The counting has to match exactly to keep instruction-count parity |
| `cpu.trigger_exception()` throws | Store the `MachineError` in the machine and return `Exit::Fault`. The return value is a one-byte enum |
| `next_execute_segment()` on out-of-segment jumps | Return `Exit::Jump`; `dispatch_simulate` switches segment and re-enters the chain with `BEGIN_BLOCK`. Returning is needed because the chain borrows the segment, and it's a slow path anyway |

Ownership: the decoded segment sits behind an `Arc` (C++ already shares
segments via `shared_ptr`). The dispatcher clones it out, so `&exec` and
`&mut machine` are disjoint borrows. The handler signature is
`fn(&DecodedSegment, usize, &mut Machine, u64, Counter) -> Exit`, where
`Counter` is the value/limit pair passed in two registers.

**Semantics follow `cpu_dispatch.cpp`.** The C++ tail-call dispatcher differs
observably from its threaded/switch dispatcher (`cpu_dispatch.cpp`), and the
oracle is the default threaded build. Where they differ, the Rust port
follows `cpu_dispatch.cpp`:

- the counter is recorded before an illegal-opcode fault (tailcall leaves it);
- an ECALL whose handler stops the machine or changes PC resumes at
  `regs.pc + 4` (tailcall resumes at `regs.pc` when it's outside the segment);
- other SYSTEM instructions retrieve both counters, and if the machine stopped
  or PC changed, continue *at* `regs.pc` (tailcall continues at `pc + 4` when
  stopping);
- the entry check is `begin <= pc < end` (tailcall: `pc + 2 <= end`).

Chains are unbounded, as in C++. They end on counter overflow, `stop()`, a
syscall that needs to leave the dispatcher, or a fault. That's only safe
because `become` is guaranteed: never write a dispatch tail call as a plain call.

Phase 1 builds this dispatcher directly with a simple decoder cache (one entry
per 2 bytes: handler + raw instruction + block length), so there's no `match`
interpreter to throw away later. Phase 4 adds bytecode specialisation, fusion,
shared segments and the `preserve_none` experiment.

### Baseline

C++ `rvlinux`, `--accurate`, no binary translation, built by
`scripts/build-cpp-reference.sh` (tailcall built with clang 21). Measured on the
dev machine on 2026-11-12:

| Variant | CoreMark iter/s | vs threaded |
|---|---|---|
| threaded (computed goto) | 2800 | — |
| tailcall (musttail + preserve_none) | 2691 | −4% |
| switch | 2012 | −28% |

All three report identical instruction counts on deterministic guests
(newlib/Rust/Zig hello-world fixtures). CoreMark picks its iteration count from
wall-clock time, so use it for speed only and never for instruction-count diffs.

Phase 1 Rust interpreter (no Phase 4 work), freestanding RV64IM collatz +
LCG loop, 206.8M instructions, same machine and date:

| Variant | Minsn/s | vs C++ threaded |
|---|---|---|
| C++ threaded | ~1100 | — |
| C++ tailcall | ~1080 | −2% |
| Rust (release) | ~708 | −36% |

## Testing

| Layer | Where | What |
|---|---|---|
| Unit | `#[cfg(test)]` in each module | Decoding, helpers, edge cases |
| Integration | `crates/libriscv/tests/` | Ports of `../libriscv/tests/unit/*.cpp` (Catch2 → `#[test]`); guests built with `testutil::build_and_load` |
| Differential | `crates/rvlinux/tests/difftest.rs`, `scripts/difftest.sh` | Same ELF under Rust and C++ `rvlinux`: stdout, exit code, instruction count and the `>>> Machine exception` line must match, also at `--fuel` timeouts. Freestanding RV64IM, RV64IMC and RV64IMC+Zcb guests, including one assembly program per dispatcher and compressed-decoder edge case; RV64IMA and RV64IMAC+Zcb for atomics; RV64IM and RV64IMC+Zcb with Zba/Zbb/Zbc/Zbs/Zicond/Zbkb for bit manipulation |
| Instruction | ports of `../libriscv/tests/instructions/` | Per-instruction checks incl. RVC (`test_rv32c.cpp` → `tests/rvc.rs`, widened to RV64) |
| ISA | `../libriscv/tests/rva23/` | Needs a full RVA23 toolchain; run by hand |
| Fuzz | `fuzz/` (Phase 5, `cargo fuzz`, nightly) | Port of `../libriscv/fuzz/fuzz.cpp`; any panic is a bug |

Prebuilt guests (Rust, Go, Zig, newlib, tinycc) live in
`../libriscv/tests/unit/elf`; use `testutil::cpp_fixture(name)`.

The C++ reference is always run with `--no-translate --accurate` so we compare
against the interpreter with exact instruction counts.
`scripts/build-cpp-reference.sh` builds the threaded, tailcall and switch
variants into `target/cpp-ref/<mode>/rvlinux` for benchmarking
(`testutil::reference_emulator_variant`).

## Phases

### Phase 0 — Infrastructure ✅

- [x] Cargo workspace, `forbid(unsafe_code)`, clippy/rustfmt config, CI
- [x] `Width`/`Address`, `MachineError`/`ExceptionKind`, `MachineOptions`, `Instr` formats
- [x] `testutil`: cross-compile guests (cached), run C++ reference, parse output
- [x] `rvlinux` CLI skeleton with output format contract
- [x] Difftest scaffolding (ignored until Phase 2)
- [x] Dispatch decision: pinned nightly + `become`; C++ dispatch variants built and baselined

### Phase 1 — Minimal machine (RV64I + M, freestanding guests) ✅

Exit criterion: port of the `Execute minimal machine` and freestanding cases of
`tests/unit/basic.cpp` / `micro.cpp` pass. Met; in addition, freestanding
guests difftest clean against the C++ oracle, including instruction counts at
timeouts.

- [x] ELF parsing and segment loading (`elf.hpp`, `memory_elf.cpp`), incl. `.text` narrowing and relocations
- [x] Flat arena memory (`memory.cpp`, `memory_inline.hpp`). Page attributes come with paging (Phase 5)
- [x] Registers, CPU state, PC (`registers.hpp`, `cpu.hpp`, `cpu_inline.hpp`)
- [x] Decoder cache per execute segment with C++'s block measurement, incl. FUNCBLOCK splitting and forced-invalid segment ends (`decoder_cache.cpp`, `decoded_exec_segment.hpp`)
- [x] Tail-call dispatcher with block-based instruction counting (`tailcall_dispatch.cpp`, `cpu_dispatch.cpp` semantics, `instruction_counter.hpp`)
- [x] RV64I + M handlers (`rvi_instr.cpp`, `rv64i.cpp`), FENCE/FENCE.I, CBO, Zicsr CSRs incl. FCSR and counters, Zimop
- [x] `ecall`/`ebreak`, syscall table, `stop()`, `return_value()`, minimal syscalls (`posix/minimal.cpp`)
- [x] `simulate(max_instructions)` / timeout error, `resume`, `simulate_inaccurate`, precise `simulate_precise`/`step_one`, `simulate_undecoded`
- [x] `rvlinux` runs freestanding guests (minimal syscalls + `exit_group`); freestanding difftests enabled

### Known differences from the oracle

Expected until the listed phase:

- Vector CSRs go to `on_unhandled_csr` (Phase 5).
- The oracle is built with virtual paging, the port is not yet (Phase 5).
  Accesses outside the arena fault here, where the oracle creates pages; a
  store into a read-only segment reports the address here and the page base
  in the oracle; FENCE.I in a program faults (the stale segment can't be
  rebuilt without paging, as in C++ without paging). A machine created
  without a binary has usable memory here, which matches the oracle (C++
  without paging has none). Freestanding guests have no crt0 to set `gp`,
  so with linker relaxation their globals land near address 0 or
  `-2048`: build them with `-mno-relax`.
- Arena allocation failure aborts the process (fallible zeroed allocation
  needs `unsafe`); C++ throws `OUT_OF_MEMORY`.

### Known C++ quirks kept for parity

Bugs or oddities in C++ that the port reproduces on purpose:

- C++'s tail-call and threaded dispatchers disagree observably (see
  "Dispatch"); the port follows the threaded one.
- `R_RISCV_RELATIVE` relocations write the symbol value, not base + addend.
- The precise JALR doesn't clear bit 0 of the target (an odd target faults);
  the dispatcher's JALR does.
- CSRRW on an unknown CSR falls through to CSRRS.
- Exception data for a force-ended segment end is C++'s re-encoded bytecode
  operand bits (`decode::cpp_entry_bits`); the uninitialised bytes C++
  leaves in LI/MV and several compressed encodings are zero here.
- The precise C.JALR writes RA before reading the target register, so
  `c.jalr ra` falls through to the next instruction; the dispatcher reads
  the target first and jumps.
- C.J is only decoded if 4 bytes at its target are inside the segment
  (C.BEQZ/C.BNEZ: 2 bytes), so a `c.j` to the segment's last halfword is an
  illegal instruction. C.LD takes the slow path while C.LDSP has a bytecode.
- The compressed slow path (`RV32C_BC_FUNCTION`) doesn't check for a stale
  segment afterwards, unlike the 4-byte one.
- A misaligned SC fails (`rd = 1`) without faulting and, unlike an
  aligned SC, keeps the reservation; a misaligned LR raises
  `DEADLOCK_REACHED` with data 0.
- Not reproduced: C++ registers an execute segment before decoding it and
  leaves it registered if decoding throws; here it's registered only when
  complete.

### Phase 2 — Linux userspace parity

Exit criterion: all difftests pass, including Rust/Go/Zig/newlib fixtures.

- [x] C extension (`rvc.hpp`, `rvc_instr.cpp`), Zcb; FP register file (raw bits) for C.FLD/C.FSD
- [x] A extension (`rva.hpp`, `rva_instr.cpp`)
- [ ] F/D (+Zfh) (`rvfd.hpp`, `rvf_instr.cpp`, `fp16.hpp`)
- [x] Zba/Zbb/Zbc/Zbs/Zicond and the RV64 part of Zbkb (in `rvi_instr.cpp`)
- [ ] `setup_linux` stack/auxv/argv/env (`machine.cpp`, `util/auxvec.hpp`)
- [ ] mmap/brk allocator (`memory_mmap.cpp`, `mmap_cache.hpp`)
- [ ] Linux syscalls incl. sandboxed fd table (`linux/system_calls.cpp`, `linux/syscalls_*.cpp`, `posix/filedesc.hpp`)
- [ ] newlib and minimal syscalls (`posix/minimal.cpp`, `machine_defaults.cpp`)
- [ ] Signals (`posix/signals.cpp`)
- [ ] `rvlinux` parity for the flags above; un-ignore difftests

### Phase 3 — Embedding API

- [ ] `install_syscall_handler`, `sysargs`, `set_result(_or_error)`
- [ ] `address_of` (ELF symbols), `vmcall`, `preempt`, call-depth handling (`machine_vmcall.hpp`, `machine_inline.hpp`)
- [ ] `PreparedCall` (`prepared_call.hpp`)
- [ ] Guest memory views, `copy_to/from_guest`, `memstring`, `stack_push`
- [ ] Printer / stdin / rdtime hooks
- [ ] Fork with CoW, `reset()` (`memory.cpp`)
- [ ] Serialize/deserialize (`serialize.cpp`)
- [ ] Ports of `tests/unit/{custom,dynamic,exceptions,examples,checksum,crc32c,pcrel,png,rvbuffer}.cpp`

### Phase 4 — Performance

- [ ] Benchmark harness: coremark/dhrystone/STREAM vs `target/cpp-ref/{tailcall,threaded}`
- [ ] Specialised bytecodes (`decode_bytecodes.cpp`, `bytecode_impl.cpp`, `threaded_bytecodes.hpp`)
- [ ] `extern "rust-preserve-none"` handlers; keep if faster
- [ ] Instruction fusion (`rvi_fuse.cpp`)
- [ ] Shared execute segments across machines (`decoded_exec_segment.hpp`, `shared_rodata.cpp`)
- [ ] Bounds-check elimination audit (`cargo asm`)

### Phase 5 — Remaining features

- [ ] V extension (`rvv.hpp`, `rvv_instr.cpp`, `rvv_registers.hpp`)
- [ ] POSIX threads / native threads (`threads.hpp`, `posix/threads.cpp`, `native_threads.cpp`)
- [ ] Native heap and libc fast paths (`native_heap.hpp`, `native_libc.cpp`)
- [ ] Guest datatypes (`guest/`, `guest_datatypes.hpp`)
- [ ] Dynamic calls (`dyncall.hpp`)
- [ ] Disassembler (`rv_printer.hpp`, `rvv_printer.hpp`, `rv_disassembly.hpp`)
- [ ] CLI debugger (`debug.cpp`, `debug.hpp`)
- [ ] GDB RSP server (`rsp_server.hpp`, `linux/rsp_server.hpp`)
- [ ] Virtual paging backend (`memory_helpers_paging.hpp`, `memory_inline_pages.hpp`, `cached_address.hpp`)
- [ ] Memory traps (`RISCV_MEMORY_TRAPS`)
- [ ] Sockets, epoll, poll, select (`posix/socket_calls.cpp`, `linux/syscalls_{epoll,poll,select}.cpp`)
- [ ] Fuzzing harness

## File mapping

`lib/libriscv/` → `crates/libriscv/src/`. Rust modules are created when their
phase starts; names below are the intended targets.

| C++ | Rust | Phase | Status |
|---|---|---|---|
| `types.hpp` | `error.rs` | 0 | ✅ |
| `common.hpp` (`MachineOptions`) | `options.rs` | 0 | ✅ (interpreter subset) |
| `rv32i_instr.hpp` | `instruction.rs` | 0 | ✅ |
| `elf.hpp`, `memory_elf.cpp` | `elf.rs`, `memory/loader.rs` | 1 | ✅ (symbol listing/lookup for backtraces: Phase 3) |
| `memory.hpp`, `memory.cpp`, `memory_inline.hpp`, `memory_rw.cpp`, `page.hpp` | `memory/` | 1 | ✅ flat arena (`page.hpp`: Phase 5) |
| `memory_mmap.cpp`, `mmap_cache.hpp` | `memory/mmap.rs` | 2 | |
| `memory_helpers_paging.hpp`, `memory_inline_pages.hpp`, `cached_address.hpp` | `memory/paging.rs` | 5 | |
| `registers.hpp`, `rvv_registers.hpp` | `registers.rs` | 1–2 | ✅ integer, FP (raw bits) + FCSR (NaN-boxing views: F/D, vector: Phase 5) |
| `cpu.hpp`, `cpu.cpp`, `cpu_inline.hpp` | `cpu.rs` | 1 | ✅ (debugger/breakpoint helpers: Phase 5) |
| `tailcall_dispatch.cpp` | `dispatch.rs` | 1 | ✅ |
| `cpu_dispatch.cpp`, `threaded_dispatch.cpp`, `cpu_inaccurate_dispatch.cpp`, `threaded_inaccurate_dispatch.cpp` | — | — | superseded by tail-call dispatch; `cpu_dispatch.cpp` defines the observable semantics (see "Dispatch") |
| `machine.hpp`, `machine.cpp`, `machine_inline.hpp` | `machine.rs` | 1–3 | Phase 1 part ✅ (`system()`, syscalls, simulate) |
| `machine_vmcall.hpp`, `prepared_call.hpp` | `vmcall.rs` | 3 | |
| `instruction_list.hpp`, `instr_decoding.inc`, `instr_helpers.hpp`, `safe_instr_loader.hpp` | `decode.rs` | 1–2 | ✅ RV64IMAC + Zcb + B (`safe_instr_loader.hpp` is in `decoder_cache.rs`) |
| `rvi_instr.cpp`, `rv64i.cpp` | `isa/rvi.rs` | 1–2 | ✅ I + M + Zba/Zbb/Zbc/Zbs/Zicond/Zbkb (Zbkb's ZIP/UNZIP are RV32-only and unimplemented, as in C++) |
| `rvc.hpp`, `rvc_instr.cpp` | `instruction.rs` (`CInstr`), `isa/rvc.rs` | 2 | ✅ RV64 C + Zcb |
| `rva.hpp`, `rva_instr.cpp` | `isa/rva.rs` | 2 | ✅ RV64 (copying the reservation on fork/deserialize: Phase 3) |
| `rvfd.hpp`, `rvfd_util.hpp`, `rvf_instr.cpp`, `fp16.hpp` | `isa/rvfd.rs` | 2 | |
| `rvv.hpp`, `rvv_instr.cpp` | `isa/rvv.rs` | 5 | |
| `rvi_fuse.cpp` | `isa/fuse.rs` | 4 | |
| `instruction_counter.hpp` | `dispatch.rs` (`Counter`) | 1 | ✅ |
| `decoder_cache.*`, `decoded_exec_segment.hpp` | `decoder_cache.rs` | 1, 4 | Phase 1 part ✅ (shared segments: Phase 4) |
| `decode_bytecodes.cpp`, `threaded_rewriter.cpp` (handler selection, target checks) | `decode.rs` (`fast_handler`, `cpp_entry_bits`) | 1–2 | ✅ RV64IMAC + Zcb + B |
| `decode_bytecodes.cpp`, `bytecode_*.cpp`, `threaded_bytecodes.hpp`, `threaded_bytecode_array.hpp`, `threaded_rewriter.cpp` | `decoder_cache/bytecodes.rs` | 4 | |
| `shared_rodata.*` | `memory/shared.rs` | 4 | |
| `machine_defaults.cpp`, `posix/minimal.cpp` | `machine.rs`, `syscalls/minimal.rs` | 1–2 | ✅ |
| `linux/system_calls.cpp`, `linux/syscalls_*.cpp` | `syscalls/linux/` | 2, 5 | |
| `posix/filedesc.hpp` | `syscalls/fds.rs` | 2 | |
| `posix/signals.*` | `signals.rs` | 2 | |
| `posix/socket_calls.cpp` | `syscalls/socket.rs` | 5 | |
| `posix/threads.cpp`, `threads.hpp`, `native_threads.cpp` | `threads.rs` | 5 | |
| `native_heap.hpp`, `native_libc.cpp` | `native/` | 5 | |
| `serialize.cpp` | `serialize.rs` | 3 | |
| `debug.*` | `debug.rs` | 5 | |
| `rv_printer.hpp`, `rvv_printer.hpp`, `rv_disassembly.hpp` | `disasm/` | 5 | |
| `rsp_server.hpp`, `linux/rsp_server.hpp` | `rsp.rs` | 5 | |
| `dyncall.hpp` | `dyncall.rs` | 5 | |
| `guest/`, `guest_datatypes.hpp` | `guest/` | 5 | |
| `livepatch.hpp` | — | — | evaluate (mostly JIT-related) |
| `util/auxvec.hpp` | `machine.rs` | 2 | |
| `util/crc32*.{hpp,cpp}` | `util/crc32.rs` | 3 | |
| `util/buffer.hpp` | `memory/buffer.rs` | 3 | |
| `util/function.hpp`, `util/threadpool.h`, `util/load_binary_file.hpp` | `std` | — | not needed |
| `emulator/src/main.cpp` | `crates/rvlinux` | 0–2 | freestanding guests, incl. the main-thread exit of `setup_posix_threads` (Linux setup: Phase 2) |
| `tr_*`, `asmjit/`, `amd64/`, `win32/`, `rv128i*`, `rva128_instr.cpp` | — | — | out of scope |
