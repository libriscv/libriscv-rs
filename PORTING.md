# Porting libriscv to safe Rust

This is the plan and progress tracker for the port. Update the status column
in the same commit that changes it.

**Reference:** C++ libriscv at `../libriscv` (override with `$LIBRISCV_CPP`),
tracking upstream commit `e806d46046fb4059ab6e605255852b9335386de2` (v1.11+).
When the C++ side moves, bump this hash and the ones in
`.github/workflows/ci.yml` and `fuzzing.yml` together.

## Goals

1. **Safe:** `#![forbid(unsafe_code)]` in every crate, with one exception:
   `libriscv-jit-rt`, the executable-memory runtime of the optional JIT (see
   "JIT"). `libriscv` itself forbids unsafe code with or without the JIT.
   Prefer zero runtime dependencies for `libriscv` (the `jit` feature adds
   Cranelift). Nightly *language features* are fine (see Toolchain); unsafe
   is not.
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
Zbs Zicond Zbkb Zfh(min) Zcb Zimop/Zcmop and V; ELF loading; flat arena memory
with virtual paging; dynamic executables (through `ld.so`); Linux, newlib and minimal syscall sets; vmcall/preempt;
fork/CoW; serialization; threads; native heap/libc; debugger, disassembler and
GDB stub.

**Out of scope** (inherently need `unsafe` or native code generation):

| C++ component | Files | Why |
|---|---|---|
| Binary translation | `tr_*.cpp`, `tr_*.hpp` | Compiles C and `dlopen`s it |
| libtcc JIT | `tr_tcc.cpp` | Same |
| asmjit JIT | `asmjit/` | Replaced by the Cranelift JIT (see "JIT"), not ported |
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

- **Memory representation.** The C++ flat read-write arena (`Vec<u8>`,
  zero-allocated so the OS commits it lazily, plus one page of padding so a
  wide access at the last address needs no size check). Guest loads and
  stores check it first with the two compares of `memory_inline.hpp`: reads
  in `[0x1000, arena_size)`, writes in `[initial_rodata_end, arena_size)`.
  (Phase 1 had only this; see "Virtual paging" for what replaced the
  "protection fault otherwise".)

Decided in Phase 5 (moved up, for dynamic executables):

- **Virtual paging: always on, behind the arena.** The oracle is built with
  `RISCV_FLAT_RW_ARENA` *and* `RISCV_VIRTUAL_PAGING`, so that is the only
  configuration (no feature, no enum of backends). The arena fast path is
  unchanged; what used to be a protection fault is now a `#[cold]` call
  into the page table. See "Virtual paging".
- **Machine shape.** `Machine { cpu, memory, .. }` is not generic. System
  call handlers are `fn(&mut Machine) -> Result<()>` in a per-machine table
  (C++'s is per process). Userdata (C++'s `void*` user pointer) is an
  owned `Box<dyn Any + Send + Sync>`, type-checked on access
  (`set_userdata`, `userdata::<T>()`, `userdata_mut`, `take_userdata`);
  an `Arc` in it shares state with the host like the C++ pointer. Forks
  don't get it, as in C++. (The generic `Machine<U>` recommended below
  wasn't worth making every machine type generic: userdata is only read
  from host callbacks, never on the dispatch path.)
- **Custom instructions.** C++'s process-wide
  `CPU::on_unimplemented_instruction` is a per-machine hook,
  `set_on_unimplemented_instruction(fn(Instr) -> Option<CustomInstruction>)`,
  kept by forks. The decoder's final fall-through (and only it, as in
  C++) is the `isa::rvi::Custom` op, which asks the hook when it
  *executes* rather than when the entry is decoded, so decoded segments
  shared between machines don't depend on any machine's hook. There is
  no disassembler yet, so a custom instruction has no printer.
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
- **Threads/fork.** `Machine` is `Send + Sync` (checked at compile time),
  so a machine shared between threads can be forked from. Fork shares
  pages via `Arc` (decided in Phase 3, see "Fork"). The only global state
  is C++'s: the process-wide cache of shared execute segments (see
  "Shared execute segments").

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
| `INSTRUCTION(bc, name)`: `static TcoRet<W> name(DecoderData<W>* d, DecodedExecuteSegment<W>* exec, CPU<W>& cpu, addr_t pc, InstrCounter& counter)` | `extern "rust-preserve-none" fn name(exec: &DecodedSegment, i: usize, m: &mut Machine, pc: u64, counter: Counter, ins: u32) -> Exit` with one signature for all handlers |
| `MUSTTAIL return computed_opcode<W>[bc](...)` | `become (entry.handler)(..., entry.instr)`. Store the handler `fn` in the decoder entry itself, so there's no second table lookup |
| `VIEW_INSTR_AS(fi, FasterItype)` reading `d->instr` | The caller passes the entry's (re-encoded) bits as `ins`, so a handler never looks up its own entry |
| `d += 1` pointer bump | index `i + 1`; read with `exec.cache.get(i)`. That's one bounds check per instruction (a compare and a predicted branch), the price of safe Rust |
| `__attribute__((preserve_none))` | `extern "rust-preserve-none" fn` (`feature(rust_preserve_none_cc)`). Without it every handler saved and restored six callee-saved registers: it cut CoreMark's host instructions by 17% |
| `return RETURN_VALUES()`, `next_execute_segment(...)` exits | `become halt_exit(...)` etc.: exits are tail calls to cold handlers too, so a fast handler makes no calls and needs no stack frame |
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

Phase 1 built this dispatcher directly with a simple decoder cache (one entry
per 2 bytes: handler + instruction bits + block length), so there was no
`match` interpreter to throw away later. Phase 4 added the bytecodes below.

### Bytecodes

The fast path is a port of C++'s threaded bytecodes, and that's where the
interpreter's speed comes from:

- `decode::fast_handler` is `decode_bytecodes.cpp` + `threaded_rewriter.cpp`:
  it picks the bytecode and re-encodes the operands into the
  `threaded_bytecodes.hpp` formats (`FasterItype`, `FasterOpType`, ...,
  in `bytecode.rs`), doing every check that can be done at decode time:
  `rd != x0`, branch and jump targets inside the segment and aligned, shift
  amounts in range, forward branches (no counter check).
- `bytecode.rs` is `bytecode_impl.cpp`: one handler per bytecode, reading
  the pre-decoded operands. They never return a `Result`: the only fallible
  ones, loads and stores, read the flat arena inline (`Memory::arena_read`,
  which can't fail) and tail-call a cold `*_slow` handler for the page
  table and faults. C++ gets the same effect from exceptions.
- Slow paths (`RV32I_BC_FUNCTION`, `RV32C_BC_FUNCTION`, `FUNCBLOCK`) keep
  the original instruction and run the precise handler, as in C++.

Where C++ sends an instruction to a FUNCTION slow path that behaves exactly
like a bytecode, this port uses the bytecode (results, counts and faults are
the same; the rewriter's C++-exact bits are kept for fault data, see
`decode::cpp_entry_bits`):

- C.SUB, C.AND, C.SUBW, C.ADDW, C.MUL, C.SRAI and C.LUI: new compressed
  bytecodes (`c_op`, `c_srai`, `c_lui`), hot in compiled code.

C.LD (`RV32C_BC_LDD`) and SRAIW (`RV64I_BC_SRAIW`) used to be in this list:
C++ had the handlers but never selected them (and its rewriter's C.LD case
swapped the destination and base registers). C++ selects both now, with
the swap fixed, and so does the port. C.LD's slow path alone took ~10% of
the interpreter's time on CoreMark.

`rvi_fuse.cpp` isn't ported: nothing in C++ calls `try_fuse`.

`tests/bytecodes.rs` runs random programs through the bytecodes and the
precise interpreter and compares registers, memory and faults.

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

Fixed-workload CoreMark, measured with the bench before it became a
CoreMark 1.0 run (see "CoreMark 1.0" below), by wall-clock time of the
whole process. Freestanding, built with our port (`crates/testutil/guests/coremark`,
from `../coremark` or `$COREMARK_DIR`, eembc/coremark `eefc986e`), 4000
iterations, `-O2`, wall-clock time of the whole process, best of 5, on
2026-11-13, with the bytecodes (see "Bytecodes"). C++ is upstream
`e806d460` (with its C.LD and SRAIW bytecodes, which made C++ threaded ~12%
faster), asmjit built by `scripts/build-cpp-reference.sh asmjit`. All runs
validate, and all exact-counting variants report the same instruction
counts (1442120574 / 1442122541 / 1272208341):

| | rv64imc | rv64imafdc | rv64imafdc_zba_zbb_zbs |
|---|---|---|---|
| Rust interpreter | 1648 ms | 1615 ms | 1516 ms |
| Rust JIT (up front) | 313 ms | 317 ms | 311 ms |
| Rust JIT (background) | 307 ms | 308 ms | 303 ms |
| C++ threaded | 1709 ms | 1624 ms | 1630 ms |
| C++ tailcall | 1751 ms | 1806 ms | 1718 ms |
| C++ asmjit (up front) | 238 ms | 240 ms | 239 ms |
| C++ asmjit (background) | 237 ms | 239 ms | 239 ms |

The guest's clock is `rdinstret`, so its output is deterministic (the same
ELFs are difftests), and CoreMark's 10-second rule passes on a nominal
clock. asmjit's instruction count only bounds execution: `rdinstret` read
from native code can be stale, which stops CoreMark's clock and fails that
rule, so variants that don't count exactly had to be validated by their
CRCs only.

Wall-clock differences of a few percent are noise on the dev machine; by
`perf stat` (rv64imc) the Rust interpreter takes 9.26G cycles and 28.6G host
instructions, C++ threaded 9.59G cycles and 21.6G instructions. Before the
bytecodes, Rust took 2196 ms and 55.7G instructions here (C++ without
C.LD/SRAIW: 1860 ms).

CoreMark as a regular Linux executable (`testutil::build_coremark_linux`:
upstream's POSIX port, static glibc, `lp64d` only; `clock()` is
`rdinstret` so the output is deterministic), same settings and date. All
exact-counting variants report 1435905543 / 1266007356 instructions:

| | glibc rv64imafdc | glibc rv64imafdc_zba_zbb_zbs |
|---|---|---|
| Rust interpreter | 1700 ms | 1511 ms |
| Rust JIT (up front) | 2196 ms | 2191 ms |
| Rust JIT (background) | 1699 ms | 1546 ms |
| C++ threaded | 1712 ms | 1659 ms |
| C++ tailcall | 1726 ms | 1715 ms |
| C++ asmjit (up front) | 401 ms | 401 ms |
| C++ asmjit (background) | 378 ms | 374 ms |

The Rust JIT doesn't pay off here: the executable segment holds all of libc
(460 KB), and compiling it takes ~1.9 s with Cranelift, after which
CoreMark runs in ~260 ms. asmjit compiles it in ~150 ms. Lazy compilation
(see Phase 4) is what fixes this.

### CoreMark 1.0

`cargo bench -p rvlinux --features jit --bench coremark` runs real
CoreMark 1.0: the POSIX port of `$COREMARK_DIR`, static glibc, `-O3`,
`ITERATIONS=0` (`testutil::build_coremark_timed`), so CoreMark calibrates
itself to at least 10 s of real time from `clock_gettime`/`gettimeofday`
and its own Iterations/Sec is the score. All variants run `--accurate`
with `--fuel 10000000000000` (the default 30G instruction limit stops a
JIT before 10 s), the JITs compile up front, and every run must validate.
`COREMARK_ELFS` adds prebuilt ELFs as columns. The difftests keep the
deterministic fixed-iteration builds above.

The prebuilt RV64 ELFs in `../coremark` are built the same way. CoreMark's
own Iterations/Sec, one run each, same date and C++ builds:

| ELF | Rust interpreter | C++ threaded | C++ tailcall | Rust JIT | C++ asmjit |
|---|---:|---:|---:|---:|---:|
| coremark-rv64g | 2354 | 2555 | 2377 | 14682 | 18559 |
| coremark-rv64gc | 2159 | 2279 | 2220 | 15038 | 19336 |
| coremark-rv64g_b | 2608 | 2704 | 2532 | 15194 | 20103 |
| coremark.rv64gb.gcc14.2.elf | 2690 | 2884 | 2549 | 15581 | 19022 |
| coremark-rv64g_zba_zbb | 2543 | 2727 | 2540 | 15848 | 19451 |

The Rust interpreter is 4-8% behind C++ threaded on every ELF (level with
tailcall), the Rust JIT was 18-24% behind asmjit (see below for the
numbers after "Bounds checks and register pressure"). With background compilation
the JIT can arrive after CoreMark has calibrated on the interpreter, which
then picks too few iterations and fails the 10-second rule; the Rust JIT
column is up front where that happened (rv64gc, rv64g_zba_zbb; same score).

After the JIT changes in "Bounds checks and register pressure" (JIT
section), 2026-11-13, Rust JIT up front against C++ asmjit up front,
interleaved runs (`perf stat` cycles per CoreMark iteration are steadier
than the score while other work loads the machine):

| ELF | Rust JIT | C++ asmjit | |
|---|---:|---:|---:|
| coremark-rv64g | 18015 | 18658 | −3.4% |
| coremark-rv64gc | 18046 | 19276 | −6.4% |
| coremark-rv64g_b | 19365-19822 | 19816-19823 | −2% to 0% |
| coremark.rv64gb.gcc14.2.elf | 17723 | 18931 | −6.4% |
| coremark-rv64g_zba_zbb | 19522 | 19411 | +0.6% |

On rv64g_b the Rust JIT now runs 6% fewer host instructions per iteration
than asmjit (1.56M vs 1.67M) at 1-3% more cycles; on rv64gc and the
gcc14.2 ELF it still runs more instructions than asmjit (not yet looked
at). The remaining difference elsewhere is mostly register allocation:
guest registers live across a whole region (loaded at entry, stored at
every exit, as in asmjit), and in high-pressure loops regalloc2 keeps
loop-carried ones in spill slots. Guest calls are still tail calls with
returns through the entry table, where asmjit makes native calls
(`emit_direct_call`) whose returns the host predicts.

## JIT

The optional `jit` feature (`MachineOptions::jit`, `rvlinux --jit`) is the
counterpart of C++'s asmjit backend, built on Cranelift instead of porting
`asmjit/`. Code: `crates/libriscv/src/jit/` (`classify.rs`: instruction
selection, `region.rs`: segment analysis and region formation, `emit.rs`:
Cranelift IR, linking) and `crates/libriscv-jit-rt`.

**Unsafe boundary.** Generating code needs no `unsafe`; mapping it executable
and calling it does. That lives in `libriscv-jit-rt`, a ~200-line crate with
its own lint table (`unsafe_code = "deny"`, every block documented): it
copies the linked image into an `mmap` (`memmap2`), validates the entry
table and the guest-memory windows against the real arena length, and
enters code through one `transmute`d trampoline pointer. `libriscv` stays
`forbid(unsafe_code)`. The generated code's own memory safety (the inlined
arena bounds checks, or the address mask, see "N-bit address space") is
the trust boundary; the rt crate documents the contract the code generator
must keep. Besides other region functions, native code only calls
Cranelift's rounding and FMA libcalls, resolved to plain Rust helpers in
`emit.rs` (only used on hosts without SSE4.1 or FMA).

**Design** (mirrors `aj_translate.cpp`/`aj_emit.cpp`):

- Execute segments are compiled when created, or with
  `MachineOptions::jit_background` (`rvlinux`'s default, as in C++) on a
  background thread while the interpreter runs them (see "Background
  compilation" below). Compilable instructions are grouped into regions that grow along fall-through and
  direct branches (loops stay inside), stopping at calls, indirect jumps,
  uncompiled instructions and other regions. One Cranelift function per
  region, `tail` calling convention, signature `(state, pc, counter)`.
- Region-to-region transfers are native tail calls (`return_call`; indirect
  jumps via the entry table and `return_call_indirect`), so calls, returns
  and loops never go back through the interpreter, and the stack doesn't
  grow.
- Guest registers (integer and FP, the latter as raw 64-bit patterns) live
  in SSA variables, loaded on function entry and stored on exit; `x0` is
  the constant zero (C++ asmjit's `get(0)`; nothing writes it, the
  register file included). Arena loads/stores are inlined with `Memory`'s
  exact range checks, against windows that are constants in the code (see
  "Bounds checks and register pressure").
  An access outside them leaves native code at that instruction
  (`EXIT_SLOW`): the interpreter runs it through the page table and
  finishes the (already counted) block.
- Decoder entries at block starts that native code can be entered at get the
  `dispatch::jit_enter` handler (C++ patches decoder entries the same way).
  Valid dyncall words (`dyncall.rs`, C++ `dyncall.hpp`) are compiled as
  counted calls, like asmjit's `emit_dyncall`/`execute_counted_dyncall`:
  native code leaves with `EXIT_DYNCALL` (all registers stored, block
  counted), `jit_enter` runs the machine's custom instruction handler
  with the exact PC, counter and limit visible, then halts if the handler
  stopped or penalized the machine (at the next instruction, with the
  count up to the dyncall), leaves a segment the handler made stale, or
  re-enters native code at the next instruction, which region formation
  always makes an entry point. A fault is reported at the dyncall's own
  PC (the interpreter reports the block's). The interpreter runs dyncalls
  like any custom instruction, without publishing the counter, as C++.
  Native code exits to the interpreter on anything it doesn't compile
  (system calls, CSRs incl. FCSR, atomics, Zfa, Zfhmin, FENCE.I, CBO,
  CLMUL), on faults and at the instruction limit, with exactly the PC,
  registers and counter the interpreter would have. FUNCBLOCK entries
  (the decoder's split of over-long blocks) are compiled: the instruction,
  then `NEXT_BLOCK`'s limit check and the next block's count, as
  `dispatch::funcblock` does. (They used to exit, which cost ~5% of
  CoreMark in `jit_enter` and the interpreter.)
- F and D (not in the `debug-checks` build, which NaN-boxes): Cranelift's
  IEEE operations as they are, whatever is fastest. For non-NaN inputs
  that's the interpreter's result (both round to nearest on the host, and
  ignore the instruction's rounding mode, as C++ does without FCSR
  emulation); which NaN comes out is the host's (NaN payloads belong to
  FCSR emulation; C++ asmjit canonicalises in FMIN/FMAX and FCVT.S.D).
  FMIN/FMAX with one NaN operand return the other operand (a number, as
  C's `fmin` requires). FCVT to integers is exact:
  the instruction's rounding mode (DYN reads FCSR.frm, which native code
  can't change), saturation, NaN to the maximum, like C++'s helper call. A
  single-precision write keeps the upper half of the register (no
  NaN-boxing), like the interpreter.
- Instruction selection follows `decode::fast_handler`, not the precise
  decoder, quirks included.
  Counting follows the dispatcher: block counts on block entry, checks at the
  same block ends, and mid-block exits pre-subtract the target's count.
- Stale segments are interpreted (native code doesn't check staleness after
  slow-path instructions, and nothing it runs can make a segment stale).
- x86-64 only for now: AArch64 needs an instruction-cache flush in the rt
  crate. Cranelift's x64 tail calls need `preserve_frame_pointers`.

**Bounds checks and register pressure** (2026-11-13; the JIT was 18-24%
behind asmjit on CoreMark 1.0, `--nbit-address-space` closed most of it,
so the checks were the cost). Each region function used to keep the arena
pointer, both windows' base and length and the instruction limit live in
registers throughout. That forced spills: each check reloaded two values
from the stack, and the exit arguments were built on the fast path. Now:

- The windows are constants in the code (C++ `aj_machine_info`'s baked
  arena bounds): a check is `lea`/`cmp`/`jae`. The layout
  (`libriscv_jit_rt::MemoryLayout`: N-bit mask or windows) is part of the
  shared segment key, and `CodeMemory::enter` refuses code whose windows
  aren't inside the machine's (the interpreter runs it then). The windows
  are final before any execute segment is created (`loader.rs`).
- Each exit (slow access, limit, misaligned jump) is a cold block of its
  own, which materialises the exit kind and PC.
- The instruction limit is read from the state block at each check (as
  asmjit compares against memory), not kept in a register.
- One check covers several accesses: the first access through a base
  register checks the span of offsets of the accesses through it up to the
  end of its straight-line run (a label, a control-flow instruction, a
  FUNCBLOCK) or until the register is written. The interpreter checks only
  an access's start address, so the span check passes exactly when all of
  the accesses' checks would; when it fails, native code leaves at the
  first access and the interpreter finishes the block, as for any slow
  access. Results and counts stay the interpreter's, unlike C++'s binary
  translator, which elides checks within the arena's over-allocation of
  an earlier one and so lets accesses at the window edges through. When
  the write window lies inside the read window (a loaded program), a store
  check also covers the loads whose offsets lie inside the stores' span:
  they lie between two writable addresses. `tests/jit.rs`
  (`merged_bounds_checks_match_the_interpreter`, which also runs a minimal
  ELF so the write window starts at the end of read-only data) sweeps base
  registers across the window edges.

Tried and dropped: making the fall-through of an in-region conditional
branch a plain label instead of an entry (asmjit's leaders aren't these),
so checks could carry across not-taken branches. It increased host
instructions by 2% and branch misses by 12% on CoreMark.

**Background compilation** (C++ `asmjit_background_callback` +
`livepatch.hpp`). C++ compiles on an embedder thread, patches a *copy* of
the decoder cache and then live-patches the running one so that threads
executing it switch over at their next block. Patching decoder entries
that another thread may be executing needs shared mutable handlers, which
safe Rust can't do without slowing every dispatch. Instead:

- A `std::thread` compiles from an `Arc` of the segment (compilation only
  reads it) and publishes a patched copy *in the segment itself*
  (`DecodedSegment::finish_background_jit`, a `OnceLock`). The
  compilation belongs to the segment, not to the machine that started it:
  every machine holding the segment (the creator, its forks, machines
  sharing it through the process-wide cache) gets the native code, as
  with C++'s live-patching of the shared segment. Nobody has to wait for
  it, before forking or otherwise.
- While one of a machine's segments has an outstanding compilation
  (running, or finished but not switched to), `dispatch_simulate` runs in
  slices of `jit_background_slice` instructions (default 2^20). Between
  slices the machine switches its segments, and the current one, to
  their published copies.
- Slices end at the dispatcher's own instruction-limit checks, and the next
  slice begins the block there, exactly what the uninterrupted run does, so
  results and counts are the same whenever the code arrives. System calls
  see the real limit (the slice end is translated in and out), a `stop()`
  or a new limit from a handler ends slicing, and the PC is only published
  where an uninterrupted run would publish it (it's observable after a
  fault). `jit::tests` checks all of this with a compilation that never
  finishes.
- Results for segments that went stale meanwhile are dropped (the copy
  shares the original's stale flag, so marking either marks both).
  `Machine::wait_for_jit` blocks until the compilations of the machine's
  segments have finished, whoever started them. A compilation that
  panics publishes "nothing compiled", so nobody waits forever.

**Testing.** The difftests run every guest a second time with `--jit` when
`rvlinux` is built with the feature (`cargo test --workspace --features
libriscv/jit,rvlinux/jit`). `crates/libriscv/tests/jit.rs` compares complete
machine state (registers, PC, counter, memory, error) between JIT and
interpreter for hand-written and random programs at ~50 instruction limits
each, and the same with background compilation (where the code arrives at
an arbitrary point); `LIBRISCV_JIT_FUZZ=<programs>[:<seed>]` runs longer
sessions (400k programs clean on 2026-11-12). FP programs are straight-line
code with a block end after every instruction, compared exactly up to the
first NaN in an FP register (then with any NaN matching any NaN, once);
the integer programs also contain the FP instructions that only move bits
(loads, stores, moves, sign injection, FCLASS, compares, integer
conversions), compared exactly. With the feature, the difftests run
`--jit` both with background compilation and with `--no-background`.

**N-bit address space** (C++ `translate_automatic_nbit_address_space`,
`MachineOptions::jit_automatic_nbit_address_space`, `rvlinux
--nbit-address-space`): with a power-of-two arena, native loads and stores
use `arena + (addr & (arena_size - 1))` without any check (a 32-bit mask
is a zero extension), like asmjit's `arena_index`. Native code then reads
and writes anywhere in the arena, read-only data and page 0 included, and
addresses outside it wrap instead of going to the page table; the
interpreter is unchanged. Other arena sizes stay checked (with a message
under `verbose_loader`; `rvlinux`'s default 4000 MiB isn't a power of two,
`-m 4096` is). The mask is a constant in the code; `CodeMemory` records it
and refuses to enter code whose mask plus the widest access doesn't fit in
the arena (which has a padding page for exactly that). Tests:
`tests/nbit_arena.rs` (port of `nbit_arena.cpp`) and the `nbit_*` cases
in `tests/jit.rs`.

**Performance** (2026-11-12, freestanding CoreMark built from
`../coremark/barebones`, 4000 iterations, `--accurate`, best of 3):

| | rv64imc | rv64imc_zba_zbb_zbs |
|---|---|---|
| Rust interpreter | 2081 ms | 1866 ms |
| Rust JIT | 252 ms | 246 ms |
| C++ threaded | 1876 ms | 1792 ms |
| C++ asmjit | 239 ms | 244 ms |

Execution is within ~5% of asmjit, and the JIT's instruction counts are
exact (asmjit's are off by a few). Cranelift compiles more slowly: compiling
the segment up front takes ~60 ms of CoreMark's 0.34 s wall time (asmjit:
0.26 s total). Next steps: background or lazy compilation, register passing
across tail calls, AArch64 (F/D came later). Background compilation (added later, see
Baseline for current numbers) saves ~5-10 ms on CoreMark, the same as
asmjit's: during the compile, the interpreter only gets through ~1/7 of
what native code would have.

## Virtual paging

`memory/paging.rs` and `memory/page.rs` port the paging half of
`memory.cpp`, `memory_rw.cpp`, `memory_helpers_paging.hpp`,
`memory_inline_pages.hpp` and `page.hpp`.

- **Page table:** `BTreeMap<page number, slot>` into a `Vec` of slots
  holding `(page number, Page)`; a page keeps its slot while it is in the
  table. A `Page` is attributes
  (`PageAttributes`: read/write/exec, copy-on-write, `dont_fork`, a user
  byte) plus where its bytes are: `Arena` (C++: a non-owning page loaned
  from the arena), `Owned(Box<[u8; 4096]>)`, or one of the shared static
  pages, `Zero` (C++ `cow_page`/`guard_page`) and `Host` (C++
  `host_page`: the exit function and signal trampoline). Owned bytes are
  an `Arc<[u8; 4096]>`, shared with forks (`Loaned` on their side) until
  one side writes: a write goes through `Arc::make_mut`, so no write ever
  reaches bytes another page shows, and there is no `unsafe` and no
  lifetime juggling. Range operations (`free_pages`, attribute changes and
  discards of huge ranges) are `BTreeMap::range` walks.
- **Page cache** (C++ `m_rd_cache`/`m_wr_cache`, `cached_readable_page`,
  `cached_writable_page`): 512 direct-mapped entries (`pageno % 512`),
  each the slot of the page last looked up there, as a relaxed
  `AtomicUsize` (filled through `&self`, like C++'s `mutable` caches,
  while keeping `Memory` `Sync`). Every lookup tries it before the B-tree.
  A hit checks the slot's page number, and loads/stores check the page's
  read/write permission, so the cache can't go stale and nothing
  invalidates it (C++ caches page pointers and invalidates them at every
  CoW, protection change, discard and free, a recurring source of bugs).
  C++ has one entry for reads and one for writes in front of a hash map;
  here a miss costs a B-tree lookup, so the cache is larger, and a guest
  can't push lookups past the B-tree's logarithm (a hash keyed by guest
  page numbers could be flooded). Without an arena (forks), every load
  and store takes this path; see "Fork" for numbers.
- **Arena pages:** C++ creates an arena page table entry whenever a slow
  path first touches the page. Here only arena pages with non-default
  attributes get an entry; a bitmap (one bit per arena page) records which
  ones C++ would have, so `pages_active()` (and `getrusage`'s `ru_maxrss`,
  and `memdiscard`'s choice between its bulk and page-by-page paths) are
  exactly C++'s.
- **Semantics:** guest loads/stores in the arena windows ignore page
  attributes, as in C++; everything else goes through the page table. Host
  accessors (`memcpy`, `memset`, `memcpy_out`, `memstring_bytes`,
  `memcmp`, `gather_buffers`, `gather_writable_buffers`) always do, so
  system calls respect page attributes inside the arena too. `memview`
  and `memview_mut` take the arena window, else a single gathered buffer
  (`OUT_OF_MEMORY` if the range isn't contiguous on the host).
- **Page faults:** a missing page outside the arena reads as the zero page
  and is created by the first write, until the owned pages reach half the
  memory limit with an arena (C++ `anywhere_pages`), the whole limit
  without one; then `OUT_OF_MEMORY`. Attribute-only pages (from
  `mmap`/`mprotect`) are bounded by 64 × the memory limit in pages. A
  writable attribute-only page is the zero page, copy-on-write.
- **Execute segments from pages** (`CPU::next_execute_segment`): jumping
  outside every known segment looks at the page; if it's executable, all
  adjacent executable pages are copied out and decoded as a new segment
  (at most `memory_max` bytes, see the differences). FENCE.I and
  `riscv_flush_icache` mark segments stale, which then get rebuilt the
  same way. The host code page is an execute-only page, not a segment
  made at load time.
- **ELF loading:** segments are copied in through the page table (so they
  can be anywhere, e.g. `-Ttext-segment` above the arena) and get their
  program header's attributes (`protect_segments`), or RWX.
- **JIT:** native code inlines the arena window checks only. Anything else
  exits to the interpreter at that instruction (`EXIT_SLOW`), which runs
  it through the page table and finishes the block with the interpreter's
  handlers (the block is already counted), so results and counts don't
  depend on the JIT. (With an N-bit address space, native code never
  reaches the page table: see "JIT".)

Dynamic executables: `rvlinux` loads the program's `PT_INTERP` (falling
back to `/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1`) as the
machine's binary, with argv `[interpreter, program, args...]`, and lets the
guest open the program and the allow-listed libraries in
`/lib/riscv64-linux-gnu/` (mapped to `/usr/riscv64-linux-gnu/lib/`), as C++
`rvlinux` does. `ld.so` maps the program and libraries with
`mmap`/`mprotect`, and their code becomes execute segments when first
jumped to. Instruction counts match the oracle exactly, also for `dlopen`.

Not ported yet: custom page fault/read/write handlers and execute fault
handlers (embedding API, Phase 3), memory traps (`RISCV_MEMORY_TRAPS`,
Phase 5) and `insert_non_owned_memory`.

## Testing

| Layer | Where | What |
|---|---|---|
| Unit | `#[cfg(test)]` in each module | Decoding, helpers, edge cases |
| Integration | `crates/libriscv/tests/` | Ports of `../libriscv/tests/unit/*.cpp` (Catch2 → `#[test]`); guests built with `testutil::build_and_load`. `linux.rs`: `security.cpp`, Linux cases of `protections.cpp`, threads, signals, the file sandbox, and every system call with random arguments. `paging.rs`: the memory cases of `protections.cpp`, `va_execute.cpp` (code on mprotected and high pages, FENCE.I/`riscv_flush_icache`, constant pools) and `dynamic.cpp` (PIE). `vmcall.rs`: `vmcall.cpp` (the fork case included) and preemption (restoring, clobbering, errors, shared budget, nested under background JIT). `brutal.rs`: `brutal.cpp` (forks, stepping a few instructions at a time, copy-on-write and page cache cases) plus fork snapshots, forks of forks, forking from threads and fork page accounting |
| Differential | `crates/rvlinux/tests/difftest.rs`, `scripts/difftest.sh` | Same ELF under Rust and C++ `rvlinux`: stdout, exit code, instruction count and the `>>> Machine exception` line must match, also at `--fuel` timeouts. Static and dynamic glibc C and C++ programs (stdio, malloc, threads incl. `test_threads.cpp`, condvars, abort, a fault with a signal handler, timeouts; dynamic ones run under `ld.so`), code generated at runtime (mprotect, FENCE.I, pages far outside the arena, `dlopen`, an execute fault), `getrusage`'s page accounting, the prebuilt Rust/Zig/newlib fixtures (Go: output only, its count varies under the oracle too) and glibc CoreMark. Freestanding RV64IM, RV64IMC and RV64IMC+Zcb guests, including one assembly program per dispatcher and compressed-decoder edge case; RV64IMA and RV64IMAC+Zcb for atomics; RV64IM and RV64IMC+Zcb with Zba/Zbb/Zbc/Zbs/Zicond/Zbkb for bit manipulation; RV64IMFD and RV64IMAFDC+Zcb with Zfhmin/Zfa for floating point; CoreMark (soft-float, F/D, F/D+B), whose clock is the instruction counter so its whole output is compared. `rvlinux --call` (a VM call after the program exits). Each scenario is one test per Rust configuration (`<name>::interp`, `::jit`, `::jit_upfront`), and long fuel loops are split, so the harness runs them concurrently |
| Instruction | ports of `../libriscv/tests/instructions/` | Per-instruction checks incl. RVC (`test_rv32c.cpp` → `tests/rvc.rs`, widened to RV64) |
| ISA | `../libriscv/tests/rva23/` | Needs a full RVA23 toolchain; run by hand |
| Fuzz | `fuzz/` (`cargo fuzz`, a separate workspace; see `fuzz/README.md`) | Port of `../libriscv/fuzz/fuzz.cpp`: `vmfuzzer` (instruction set), `elffuzzer` (ELF loader + Linux system calls, seeded with the prebuilt RV64 ELFs), `syshelperfuzzer` (system call memory helpers). Built with debug assertions and overflow checks; any panic is a bug. `FUZZ_NAT` waits for the native heap |

Prebuilt guests (Rust, Go, Zig, newlib, tinycc) live in
`../libriscv/tests/unit/elf`; use `testutil::cpp_fixture(name)`.

CI also tests the 4-byte decoder layout (`--no-default-features`, with and
without `jit`). The distribution's glibc is RV64GC, so tests running glibc
guests are ignored (or, for `linux.rs` and `vmcall.rs`, compiled out)
without the A, C and F features; freestanding guests are built for what
the features allow.

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
- [x] Flat arena memory (`memory.cpp`, `memory_inline.hpp`). Page attributes came with paging (Phase 5)
- [x] Registers, CPU state, PC (`registers.hpp`, `cpu.hpp`, `cpu_inline.hpp`)
- [x] Decoder cache per execute segment with C++'s block measurement, incl. FUNCBLOCK splitting and forced-invalid segment ends (`decoder_cache.cpp`, `decoded_exec_segment.hpp`)
- [x] Tail-call dispatcher with block-based instruction counting (`tailcall_dispatch.cpp`, `cpu_dispatch.cpp` semantics, `instruction_counter.hpp`)
- [x] RV64I + M handlers (`rvi_instr.cpp`, `rv64i.cpp`), FENCE/FENCE.I, CBO, Zicsr CSRs incl. FCSR and counters, Zimop
- [x] `ecall`/`ebreak`, syscall table, `stop()`, `return_value()`, minimal syscalls (`posix/minimal.cpp`)
- [x] `simulate(max_instructions)` / timeout error, `resume`, `simulate_inaccurate`, precise `simulate_precise`/`step_one`, `simulate_undecoded`
- [x] `rvlinux` runs freestanding guests (minimal syscalls + `exit_group`); freestanding difftests enabled

### Known differences from the oracle

Expected until the listed phase:

- Vector CSRs go to `on_unhandled_csr` (Phase 5), and the unit-stride
  vector loads/stores in the LOAD-FP/STORE-FP opcodes are illegal
  instructions here.
- The oracle's FMIN/FMAX result for a single *signaling* NaN operand
  depends on the C++ compiler: clang inlines `std::fmin` as `minnum` (the
  other operand, as here), while a call into glibc returns a quiet NaN. The
  difftests avoid that case.
- FMADD/FMSUB/FNMADD/FNMSUB NaN results (payload and sign) come from the
  host's `fma`: C++ calls glibc's, which uses the CPU's FMA instruction,
  and which NaN operand that returns differs between CPUs; Rust's
  `mul_add` calls compiler-builtins' software `fma`. They agree on an AMD
  Ryzen 7950X but differed on a GitHub runner. Calling glibc's `fma` would
  need `unsafe`. The float difftest compares only whether the result is NaN.
- Freestanding guests have no crt0 to set `gp`, so with linker relaxation
  their globals land near address 0 or `-2048`: build them with
  `-mno-relax`.
- Arena allocation failure aborts the process (fallible zeroed allocation
  needs `unsafe`); C++ throws `OUT_OF_MEMORY`. A `memory_max` too large to
  address at all gets no arena (paging only), like a failed C++ `mmap`.

Deliberate, permanent differences in paging:

- Page 0 is a guard page in every machine; C++ only installs it when
  loading a binary. A machine without a binary also keeps its arena fast
  path (C++ leaves the boundaries at 0 and serves the arena through the
  page table, with the same results except for page 0).
- An execute segment built from executable pages is at most `memory_max`
  bytes (`OUT_OF_MEMORY` otherwise): C++ would decode e.g. a guest's
  `mprotect(PROT_EXEC)` of 64 × `memory_max` of attribute-only pages,
  exhausting host memory.
- Slow-path accesses that cross a page boundary use both pages (C++ reads
  or writes past the end of a heap page; for arena pages, as here, the
  next arena bytes).
- Discarding the host code page gives the machine a zeroed private copy;
  C++ zeroes the shared static page, for every machine in the process.

### VM calls

`vmcall.rs` ports `machine_vmcall.hpp` and `CPU::preempt_internal`.
C++'s variadic arguments are a tuple of `VmCallArg`s (a single argument
or `()` also work), with C++'s rules per type: integers sign- or
zero-extended into `a0`.., `f32`/`f64` into `fa0`.., strings pushed with
their NUL, slices/arrays/`Vec`s of `GuestPod` pushed as their elements,
and `OnStack(pod)` for other standard-layout values (`GuestPod` replaces
C++'s raw `memcpy` of the object). Template parameters became methods:

| C++ | Rust |
|---|---|
| `vmcall(f, args...)` (`MAXI = UINT64_MAX`: `simulate_inaccurate`) | `vmcall(f, args)` |
| `vmcall<MAXI>` / `vmcall<MAXI, false>` | `vmcall_limited` / `try_vmcall` (0 or `u64::MAX`: unlimited, as in C++) |
| `preempt<true, true>` | `preempt` (registers, PC included, restored) |
| `preempt<true, false>` | `preempt_clobbering` |
| `preempt<false, StoreRegs>` | `try_preempt(.., store_regs)` |

`f` is an address or a symbol name (`CallTarget`). Preemption runs on top
of the current counter with a limit of `counter + max` and restores the
caller's limit afterwards, so the instructions count against the caller
too. `dispatch_simulate` is reentrant for preemption from system call
handlers: the decoded segment is held by the running chain's own `Arc`,
`DispatchState`'s exit fields are consumed before the chain returns, and
the background-JIT slice is saved and restored around every run (a
nested run that inherited the outer slice could stop at the outer
slice's end; `tests/vmcall.rs` `preemption_during_background_slice`).

C++ behaviour kept for parity:

- If pushing an argument fails, the error comes back with the registers
  as far as the setup got (SP lowered, RA set, the arguments so far),
  also for `preempt`, which only restores registers when the *call*
  fails. That clobbers a guest interrupted in a system call; probably a
  C++ oversight.
- Integer arguments past `a7` continue into `s2`.., FP ones past `fa7`
  into `fs2`..; past `x31`/`f31` is an `ILLEGAL_OPERATION` error here
  (C++ writes past the register file).
- A clobbering preemption from a system call handler leaves PC at the
  exit function, so the interrupted code continues 4 bytes after it, as
  after any handler that changes PC.
- `default_exit_function`: when the symbol exists, the host page (with
  the signal return trampoline) isn't installed.

### Fork

`Machine::fork_with(&self, options)` ports the forking constructor
`Machine(const Machine&, MachineOptions)` (`fork()` uses the master's own
options; C++'s default is `MachineOptions{}`). The fork copies the
registers, instruction counter and limit, threads, program layout, mmap
free list and LR/SC reservation, and shares the execute segments and the
system call table (an `Arc`, copied when either side installs a handler;
C++'s table is per process). Like C++, it gets the default printer, stdin
and rdtime, no file descriptors and no signal handlers.

Memory (`Memory::fork`, C++ `machine_loader`): every page except
`dont_fork` ones is loaned. Its slot keeps its number, so the fork clones
the page index as is and maps the slots, an `Arc` increment each;
writable pages become copy-on-write in the fork. `minimal_fork` loans
nothing (C++ then also has no page fault handler, expecting a custom one,
which isn't ported; here the default handler gives zero pages). A fork
has no arena. Forking a machine that has one with `use_memory_arena` set
is `ILLEGAL_OPERATION`, as in C++. Without it, the fork gets copies of
the arena pages C++ has in its page table (C++ loans them out of the
master's arena) and, like C++, doesn't see arena pages only guest
loads/stores touched. Machines meant for forking are created with
`use_memory_arena: false`.

Deliberate differences, all stricter than C++:

- **Forks are snapshots.** The master may run, change and be dropped
  while forks exist (C++: it must outlive its forks and mostly not
  change); a write to bytes a fork shares copies them first, on either
  side. Forks of forks work the same way.
- A loaned page made writable in a fork (`mprotect`) is copy-on-write,
  and `madvise(MADV_DONTNEED)` of a read-only loaned page zeroes a copy.
  C++ writes the master's page in both cases.

Forks share the master's execute segments, including ones still being
compiled in the background (see "Background compilation"), and segments
a fork creates later (e.g. the host exit page on its first VM call) come
from the process-wide cache, decoded once (see "Shared execute
segments").

Native code doesn't *run* in forks yet: a fork has no arena, and
`JitState::new` refuses windows that don't fit in it, so `jit_enter`
falls back to the interpreter (the same holds for the masters meant for
forking, created with `use_memory_arena: false`). C++ asmjit runs in forks
and reaches the page table from native code; the Cranelift JIT leaves
native code for every access outside the arena (`EXIT_SLOW`), so it needs
a page-table path in native code first. JIT code compiled for the
master's N-bit arena would be refused in a fork anyway.

### Shared execute segments

Port of C++ `SharedExecuteSegments` (`decoder_cache.cpp`), on by default
(`MachineOptions::use_shared_execute_segments`): machines of the process
that decode the same code at the same address share one
`Arc<DecodedSegment>`, with its native code. Keyed by address, length,
CRC32-C (`crc32c`, slicing-by-8 instead of C++'s SSE4.2 instruction,
which needs `unsafe`), execute-only, and whether and how the JIT compiles
(none, or its N-bit mask). C++ also keys on the arena boundaries, which
asmjit bakes into its code; Cranelift code reads them at run time. As in
C++, the cache entry stays locked while a segment is decoded, so
concurrent machines decode and compile it once, and a machine compiling
up front waits for a background compilation of the same segment instead
of starting its own.

Differences from C++:

- **The code is compared, not just the CRC.** CRC32-C collisions are
  trivial to construct (the unit test forges one), and C++ hands the
  registered segment to any machine whose code collides, i.e. one guest's
  machine can end up executing code another guest supplied. Here a
  collision gets its own, unshared segment.
- Stale segments are never handed out; a new one replaces them in the
  cache.
- The cache holds segments weakly: they die with the last machine using
  them (C++ `remove_if_unique` on eviction and destruction), and entries
  nobody uses are pruned when the map has doubled.

`crates/libriscv/benches/fork.rs` against the same scenario in C++
(Release, threaded dispatch; master: static glibc, 637 pages), per
operation:

| | Rust | C++ |
|---|---|---|
| fork + drop | 6.3 µs | 15.2 µs |
| fork + VM call writing 1 page | 6.4 µs | 14.9 µs |
| fork + VM call writing 64 pages | 13.2 µs | 20.8 µs |
| fork + VM call reading 2 MiB | 1.64 ms | 1.50 ms |
| random read-modify-write over 512 pages (per iteration) | 22.9 ns | 30.3 ns |

Before the page cache, the last two were 3.44 ms and 84 ns. The
copy-on-write row depends on glibc: when a dropped fork's pages coalesce
at the top of the heap, glibc trims it and the next fork faults them in
again (64 pages: 36 µs), unless its dynamic trim threshold was raised
earlier by freeing a large `mmap`ed block, as the C++ harness happens to
do. The bench does that first (`FORK_BENCH_NO_WARMUP=1` skips it).

### Linux system calls

`syscalls/linux/` follows `linux/system_calls.cpp`, with the host reached
only through `std` (no raw file descriptors, no `libc`). Where C++ hands
the guest raw host access, the port is deliberately stricter:

- `ioctl` never reaches the host (C++ forwards terminal requests in proxy
  mode): `ENOSYS`, or `EPERM` from `filter_ioctl`. `fcntl` is emulated
  (`F_GETFD`/`F_SETFD`/`F_GETFL`/`F_SETFL`/`F_DUPFD*`) instead of being
  forwarded with guest arguments.
- `openat` refuses write access (`O_WRONLY`/`O_RDWR`/`O_CREAT`/`O_TRUNC`/
  `O_APPEND`) outside proxy mode with `EACCES`; C++ opens (creating or
  truncating the host file) and only refuses the writes.
- Host paths are only looked up through the filters: `fstatat` with a path
  needs `filter_stat` (or `filter_open` without rewriting), `faccessat`
  needs `filter_open`, `readlinkat` needs `filter_readlink`; without them
  only proxy mode allows it. C++ goes to the host when no filter is set.
- `dup` returns a virtual descriptor (C++: a host number the guest can't
  use), `dup3` can't replace descriptors (`EPERM`), `pipe2` and
  `getdents64` are `ENOSYS` (a pipe nobody reads would block the host),
  reading fds 1/2 is `EBADF`, and the guest can have at most 1024 files
  open (`EMFILE`).
- `ppoll` answers from the descriptor table (stdin readable, stdout/stderr
  writable, files both, unknown `POLLNVAL`) instead of polling host fds,
  and never waits.
- `getrusage` is always the deterministic answer (`ru_maxrss` from the
  pages in use, as C++); time is `SystemTime` and monotonic clocks count
  from machine creation (C++: host uptime).
- `getcwd` checks the buffer size (`ERANGE`); `readlinkat` truncates to
  the buffer.
- Anonymous `mmap` and `madvise(DONTNEED)` zero memory without committing
  never-touched pages (C++ uses `madvise`).

System calls reach guest memory through the gather functions with C++'s
buffer counts (`read`/`pread64` 512, `write` 64, file `mmap` 256), so a
range of more scattered pages than that is `OUT_OF_MEMORY`, as in C++.

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
- C++'s fast-path FMADD bytecode is chosen for every format and does nothing
  for H and Q (funct2 2 and 3), where the precise handler raises
  `ILLEGAL_OPERATION`. FMSUB/FNMADD/FNMSUB always take the precise path.
- Without NaN-boxing (the default, like C++ without `RISCV_FCSR`), a
  single-precision result only replaces the low 32 bits of an f-register,
  and FLH/FMV.H.X zero-extend the half into the low word.
- C.J is only decoded if 4 bytes at its target are inside the segment
  (C.BEQZ/C.BNEZ: 2 bytes), so a `c.j` to the segment's last halfword is an
  illegal instruction.
- The compressed slow path (`RV32C_BC_FUNCTION`) doesn't check for a stale
  segment afterwards, unlike the 4-byte one.
- A misaligned SC fails (`rd = 1`) without faulting and, unlike an
  aligned SC, keeps the reservation; a misaligned LR raises
  `DEADLOCK_REACHED` with data 0.
- Not reproduced: C++ registers an execute segment before decoding it and
  leaves it registered if decoding throws; here it's registered only when
  complete.
- `rvlinux` keeps the NUL terminator of `PT_INTERP` in a dynamic
  executable's `argv[0]` (C++ copies the whole `p_filesz` into a
  `std::string`), so that string takes one byte more on the stack.
- Guest stores inside the arena's write window, and loads in its read
  window, ignore page attributes (`mprotect(PROT_NONE)` doesn't stop them);
  system calls and other host accesses respect them.
- `memdiscard` (`MADV_DONTNEED`, anonymous `mmap`) zeroes read-only memory
  too, and a run of whole untouched arena pages is wiped without looking at
  the attributes of the pages after the first.
- `clone` reads `ptid`/`tls`/`ctid` from `a4`-`a6`, where glibc's wrapper
  leaves them (the kernel ABI has them in `a2`-`a4`); `set_tid_address`
  truncates the address to an `int`.
- A futex wait that would deadlock (no other runnable thread) writes 8
  zero bytes at the futex address and returns 0. When the running thread
  exits or blocks and nothing is suspended, the main thread is resumed
  even if it's blocked. Exiting threads clear their TID word but don't wake
  futex waiters; futex timeouts are ignored.
- `getrandom` of more than 256 bytes returns -1 (not an errno) and charges
  20k instructions per byte; `uname` says "RISC-V C++ Emulator";
  `readlinkat` with a buffer over 1024 bytes is `ENOMEM`; `nanosleep`
  writes the clamped request back; `sigaction` reports an unset handler as
  `~0xF`.
- EBREAK isn't a block end, so a handler's `stop()` only takes effect at
  the next check (C++ `basic.cpp` accepts both 51 and 25 EBREAKs).
- The host-side `membuffer`, `memstring_view` and `try_memmove` take the
  arena fast path after checking only the arena's size, not its read or
  write windows or page attributes (`memstring_view` even below
  `RWREAD_BEGIN`), and `memcpy_from` uses the arena's write window for its
  source. `memcmp` doesn't check that the start is readable on its arena
  fast path, only the end.
- Not reproduced (C++ bug): the slow path of `memcmp(const void*,
  address_t, size_t)` compares the host byte as a signed `char`, flipping
  the result for bytes above 0x7F; here every path compares unsigned bytes
  (and returns an `Ordering`, since only `memcmp`'s sign is meaningful).
- Not reproduced (C++ bugs): `ppoll` with a NULL timeout faults in C++;
  killing a suspended thread (`tgkill`) leaves a dangling pointer in C++'s
  scheduler lists, here the thread is removed from them; a signal number
  above 64 throws `std::out_of_range`, here it's `ILLEGAL_OPERATION`.

### Phase 2 — Linux userspace parity

Exit criterion: all difftests pass, including Rust/Go/Zig/newlib fixtures.

- [x] C extension (`rvc.hpp`, `rvc_instr.cpp`), Zcb; FP register file (raw bits) for C.FLD/C.FSD
- [x] A extension (`rva.hpp`, `rva_instr.cpp`)
- [x] F/D, Zfhmin, Zfa (`rvfd.hpp`, `rvfd_util.hpp`, `rvf_instr.cpp`, `fp16.hpp`), C++'s default (non-FCSR) semantics. FCSR emulation (`fcsr` feature) not ported: C++ computes exact results in `long double`; enabling the feature is a compile error until it is
- [x] Zba/Zbb/Zbc/Zbs/Zicond and the RV64 part of Zbkb (in `rvi_instr.cpp`)
- [x] `setup_linux` stack/auxv/argv/env, `setup_argv`, `stack_push` (`machine.cpp`, `util/auxvec.hpp`)
- [x] mmap/brk allocator (`memory_mmap.cpp`, `mmap_cache.hpp`), flat-arena `memdiscard`
- [x] Linux syscalls incl. sandboxed fd table (`linux/system_calls.cpp`, `linux/syscalls_{mman,poll,select}.cpp`, `posix/filedesc.hpp`); see "Linux system calls" below. epoll/eventfd/sockets: Phase 5
- [x] newlib and minimal syscalls (`posix/minimal.cpp`, `machine_defaults.cpp`)
- [x] Signals (`posix/signals.cpp`): sigaction, sigaltstack, tkill/tgkill delivery, rt_sigreturn
- [x] POSIX threads (`threads.hpp`, `posix/threads.cpp`): clone/clone3, futex, sched_yield, exit; moved up from Phase 5
- [x] `rvlinux` parity: `setup_linux` + Linux syscalls + threads for every guest, `--sandbox`/`--proxy`/`--allow`, the C++ open/readlink filters, `run_sighandler` after a fault; Linux difftests enabled
- [x] Dynamic executables: `rvlinux` loads the interpreter (`ld.so`), code mapped at runtime becomes execute segments (virtual paging, moved up from Phase 5); dynamic difftests incl. `dlopen`

### Phase 3 — Embedding API

- [x] `install_syscall_handler`, typed `sysarg`/`sysargs`/`return_value`, `set_result(_or_error)` (`sysargs.rs`: `SysArg`/`SysResult` traits, tuples of up to 8; `ByPointer<T>` for C++'s pointer-to-struct arguments)
- [x] `address_of` (ELF symbols), `vmcall`, `preempt` (restoring and clobbering), `setup_call`, `default_exit_function`, `rvlinux --call` (`machine_vmcall.hpp`, `machine_inline.hpp`, `CPU::preempt_internal`); see "VM calls"
- [ ] Call-depth handling
- [ ] `PreparedCall` (`prepared_call.hpp`)
- [x] Guest memory views, `copy_to/from_guest`, `memstring`, `stack_push` (`memory/helpers.rs`: `Buffer`/`membuffer`, `strlen`, `memstring_view`, `memview`/`memview_mut` with `maxlen`, typed `memarray`/`memspan`/`try_memarray` as `GuestSlice`/`GuestSliceMut` over `GuestPod`, `try_memmove`, both `memcmp`s, machine-to-machine `memcpy_from`, `read_value`/`write_value`; `stack_push_value`)
- [x] Printer / stdin / rdtime hooks
- [x] Userdata (`set_userdata`/`get_userdata`), custom instructions (`on_unimplemented_instruction`); see "Architecture decisions"
- [x] Fork with CoW (`machine.cpp`, `cpu.cpp`, `memory.cpp` `machine_loader`), page caches; see "Fork". (C++ `Memory::reset()` is empty: nothing to port)
- [ ] Serialize/deserialize (`serialize.cpp`)
- [ ] Ports of `tests/unit/{dynamic,exceptions,examples,checksum,pcrel,png}.cpp` (`rvbuffer.cpp` and `custom.cpp`'s "Take custom system arguments": done, `tests/guest_memory.rs`; "Custom instruction": done, `tests/custom.rs`) (`dyncall.cpp`: done, `tests/dyncall.rs`) (`crc32c.cpp`: the CRC32-C case, in `decoder_cache.rs`) (`vmcall.cpp`: done, `tests/vmcall.rs`; `brutal.cpp`: done except memory traps, `tests/brutal.rs`)

### Phase 4 — Performance

- [x] Benchmark harness: CoreMark (`crates/rvlinux/benches/coremark.rs`) vs the C++ reference, `target/cpp-ref/{threaded,tailcall,asmjit}`; VM calls (`crates/libriscv/benches/vmcall.rs`)
- [ ] Benchmark harness: dhrystone, STREAM
- [x] Specialised bytecodes (`decode_bytecodes.cpp`, `threaded_rewriter.cpp`, `bytecode_impl.cpp`, `threaded_bytecodes.hpp`); see "Bytecodes"
- [x] `extern "rust-preserve-none"` handlers
- [x] Instruction fusion (`rvi_fuse.cpp`): nothing to port, C++ never calls `try_fuse`
- [x] Shared execute segments across machines and forks, background JIT results reaching every holder (`decoder_cache.cpp` `SharedExecuteSegments`, `util/crc32.hpp`); see "Shared execute segments"
- [ ] Shared read-only data (`shared_rodata.cpp`)
- [ ] Bounds-check elimination audit (`cargo asm`)
- [x] Cranelift JIT (`jit` feature) replacing asmjit; see "JIT". RV64IMFDC, Zcb, B (no CLMUL), Zicond; A, CSRs, Zfa and Zfhmin stay interpreted
- [x] JIT: background compilation (`jit_background`, `rvlinux -B`/`--no-background`)
- [x] JIT: F/D (FCVT to integers exact, NaNs the host's)
- [x] JIT: automatic N-bit address space (`jit_automatic_nbit_address_space`, `rvlinux --nbit-address-space`; port of `nbit_arena.cpp`)
- [ ] JIT: lazy compilation, AArch64 (icache flush in `libriscv-jit-rt`). Lazy compilation matters most for dynamic executables: `libc.so`'s whole executable mapping (1.5 MB) becomes one segment, and compiling it up front takes ~10 s (release)

### Phase 5 — Remaining features

- [ ] V extension (`rvv.hpp`, `rvv_instr.cpp`, `rvv_registers.hpp`)
- [ ] Native threads (`native_threads.cpp`); POSIX threads are done (Phase 2)
- [ ] Native heap and libc fast paths (`native_heap.hpp`, `native_libc.cpp`)
- [ ] Guest datatypes (`guest/`, `guest_datatypes.hpp`)
- [x] Dynamic calls (`dyncall.hpp`): the encoding, counted calls in the JIT (see "JIT")
- [ ] Disassembler (`rv_printer.hpp`, `rvv_printer.hpp`, `rv_disassembly.hpp`)
- [ ] CLI debugger (`debug.cpp`, `debug.hpp`)
- [ ] GDB RSP server (`rsp_server.hpp`, `linux/rsp_server.hpp`)
- [x] Virtual paging (`memory_helpers_paging.hpp`, `memory_inline_pages.hpp`, `page.hpp`, paging parts of `memory.cpp`/`memory_rw.cpp`/`cpu.cpp`); see "Virtual paging" (page caches included)
- [ ] Memory traps (`RISCV_MEMORY_TRAPS`)
- [ ] Sockets, epoll, poll, select (`posix/socket_calls.cpp`, `linux/syscalls_{epoll,poll,select}.cpp`)
- [x] Fuzzing harness (`fuzz/`; `FUZZ_NAT` waits for the native heap)

## File mapping

`lib/libriscv/` → `crates/libriscv/src/`. Rust modules are created when their
phase starts; names below are the intended targets.

| C++ | Rust | Phase | Status |
|---|---|---|---|
| `types.hpp` | `error.rs` | 0 | ✅ |
| `common.hpp` (`MachineOptions`) | `options.rs` | 0 | ✅ (interpreter subset) |
| `rv32i_instr.hpp` | `instruction.rs` | 0 | ✅ |
| `elf.hpp`, `memory_elf.cpp` | `elf.rs`, `memory/loader.rs` | 1 | ✅ (symbol listing/lookup for backtraces: Phase 3) |
| `memory.hpp`, `memory.cpp`, `memory_inline.hpp`, `memory_rw.cpp` | `memory/` | 1, 3, 5 | ✅ arena + paging + fork (serialization: Phase 3) |
| `page.hpp` | `memory/page.rs` | 5 | ✅ (memory traps: Phase 5) |
| `memory_mmap.cpp`, `mmap_cache.hpp` | `memory/mmap.rs` | 2 | ✅ (+ flat-arena `memdiscard`, `memstring`, `flush_execute_segments`) |
| `memory_helpers_paging.hpp`, `memory_inline_pages.hpp` | `memory/paging.rs`, `memory/helpers.rs` | 3, 5 | ✅ incl. page caches, fork page tables and the host-side helpers (`membuffer`, `memarray`, `memspan`, ...) |
| `cached_address.hpp` (`CachedAddress`) | — | 3 | not yet (a cached `address_of`) |
| `registers.hpp`, `rvv_registers.hpp` | `registers.rs` | 1–2 | ✅ integer, FP (raw bits) + FCSR (the `fp64reg` views are in `isa/rvfd.rs`; vector: Phase 5) |
| `cpu.hpp`, `cpu.cpp`, `cpu_inline.hpp` | `cpu.rs` | 1 | ✅ (debugger/breakpoint helpers: Phase 5) |
| `tailcall_dispatch.cpp` | `dispatch.rs` | 1, 4 | ✅ |
| `cpu_dispatch.cpp`, `threaded_dispatch.cpp`, `cpu_inaccurate_dispatch.cpp`, `threaded_inaccurate_dispatch.cpp` | — | — | superseded by tail-call dispatch; `cpu_dispatch.cpp` defines the observable semantics (see "Dispatch") |
| `machine.hpp`, `machine.cpp`, `machine_inline.hpp` | `machine.rs`, `sysargs.rs` | 1–3 | Phase 1 part ✅ (`system()`, syscalls, simulate), fork ✅, typed `sysargs`/`set_result` ✅ |
| `machine_vmcall.hpp`, `prepared_call.hpp` | `vmcall.rs` | 3 | ✅ `vmcall`/`preempt` (`PreparedCall`: not yet) |
| `instruction_list.hpp`, `instr_decoding.inc`, `instr_helpers.hpp`, `safe_instr_loader.hpp` | `decode.rs` | 1–2 | ✅ RV64IMAFDC + Zcb + B + Zfhmin/Zfa (`safe_instr_loader.hpp` is in `decoder_cache.rs`) |
| `rvi_instr.cpp`, `rv64i.cpp` | `isa/rvi.rs` | 1–2 | ✅ I + M + Zba/Zbb/Zbc/Zbs/Zicond/Zbkb (Zbkb's ZIP/UNZIP are RV32-only and unimplemented, as in C++) |
| `rvc.hpp`, `rvc_instr.cpp` | `instruction.rs` (`CInstr`), `isa/rvc.rs` | 2 | ✅ RV64 C + Zcb |
| `rva.hpp`, `rva_instr.cpp` | `isa/rva.rs` | 2 | ✅ RV64 (the reservation is copied on fork; on deserialize: Phase 3) |
| `rvfd.hpp`, `rvfd_util.hpp`, `rvf_instr.cpp`, `fp16.hpp` | `isa/rvfd.rs` | 2 | ✅ F, D, Zfhmin, Zfa (RV64), NaN-boxing under `debug-checks` (FCSR emulation: not ported) |
| `rvv.hpp`, `rvv_instr.cpp` | `isa/rvv.rs` | 5 | |
| `rvi_fuse.cpp` | — | — | dead code in C++ (`try_fuse` is never called) |
| `instruction_counter.hpp` | `dispatch.rs` (`Counter`) | 1 | ✅ |
| `decoder_cache.*`, `decoded_exec_segment.hpp` | `decoder_cache.rs` | 1, 4 | ✅ incl. shared segments (EBREAK locations, libc fast path: not yet) |
| `decode_bytecodes.cpp`, `threaded_rewriter.cpp` | `decode.rs` (`fast_handler`, `cpp_entry_bits`) | 1–4 | ✅ RV64IMAFDC + Zcb + B |
| `bytecode_impl.cpp`, `threaded_bytecodes.hpp` | `bytecode.rs` (slow-path and system bytecodes: `dispatch.rs`) | 4 | ✅ |
| `bytecode_dispatch.cpp`, `threaded_bytecode_array.hpp` | — | — | superseded by tail-call dispatch |
| `shared_rodata.*` | `memory/shared.rs` | 4 | |
| `machine_defaults.cpp`, `posix/minimal.cpp` | `machine.rs`, `syscalls/minimal.rs` | 1–2 | ✅ |
| `linux/system_calls.cpp`, `linux/syscalls_*.cpp` | `syscalls/linux/` | 2, 5 | ✅ incl. mman, poll, select (epoll, eventfd: Phase 5) |
| `posix/filedesc.hpp` | `syscalls/fds.rs` | 2 | ✅ (`std::fs::File`s, no raw fds) |
| `posix/signals.*` | `signals.rs` | 2 | ✅ |
| `posix/socket_calls.cpp` | `syscalls/socket.rs` | 5 | |
| `posix/threads.cpp`, `threads.hpp`, `native_threads.cpp` | `threads.rs` | 2, 5 | ✅ POSIX threads (native threads: Phase 5) |
| `native_heap.hpp`, `native_libc.cpp` | `native/` | 5 | |
| `serialize.cpp` | `serialize.rs` | 3 | |
| `debug.*` | `debug.rs` | 5 | |
| `rv_printer.hpp`, `rvv_printer.hpp`, `rv_disassembly.hpp` | `disasm/` | 5 | |
| `rsp_server.hpp`, `linux/rsp_server.hpp` | `rsp.rs` | 5 | |
| `dyncall.hpp` | `dyncall.rs` | 5 | ✅ (JIT side in `jit/`, see "JIT") |
| `guest/`, `guest_datatypes.hpp` | `guest/` | 5 | |
| `livepatch.hpp` | — | — | evaluate (mostly JIT-related) |
| `asmjit/aj_translate.cpp`, `aj_emit.cpp`, `aj_api.cpp` | `jit/` (Cranelift), `libriscv-jit-rt` | 4 | ✅ replaced, not ported |
| `util/auxvec.hpp`, `Machine::setup_linux`/`setup_argv` | `linux.rs` | 2 | ✅ |
| `util/crc32*.{hpp,cpp}` | `decoder_cache.rs` (`crc32c`) | 4 | ✅ CRC32-C (plain CRC32: not needed yet) |
| `util/buffer.hpp` | `memory/helpers.rs` | 3 | ✅ `Buffer` |
| `util/function.hpp`, `util/threadpool.h`, `util/load_binary_file.hpp` | `std` | — | not needed |
| `emulator/src/main.cpp` | `crates/rvlinux` | 0–2, 5 | ✅ static and dynamic Linux guests (`full_linux_guest`), `--call` (without the system call 500 symbol helper, which needs userdata); debugger, GDB: later |
| `tr_*`, `amd64/`, `win32/`, `rv128i*`, `rva128_instr.cpp` | — | — | out of scope |
