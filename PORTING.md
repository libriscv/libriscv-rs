# Porting libriscv to safe Rust

Architecture, implementation status, and validation requirements for the port.
Update status entries in the same commit as implementation changes.

**Reference:** C++ libriscv at `../libriscv` (override with `$LIBRISCV_CPP`),
tracking upstream commit `cafba3e8c29feb76a7fdd9a553f3b25014001d6e` (v1.11+)
with `scripts/cpp-reference-patches/*.patch` applied (`git apply`, in order):
the shared-library pinning (`FileDescriptors::pinned_mappings`) that
upstream master does not have. Drop a patch once the pinned commit
contains it. Update this hash together with the reference hashes in
`.github/workflows/ci.yml` and `fuzzing.yml`.

## Goals

1. **Memory safety.** `libriscv` enforces `#![forbid(unsafe_code)]` with all
   features. Unsafe operations are isolated in `libriscv-jit-rt`: executable
   memory, AOT shared-object loading, and shared guest-arena mappings.
   Embedded AOT translations require no unsafe code. Interpreter builds
   should have no runtime dependencies; `jit` adds Cranelift.
2. **Guest faults return errors.** Guest input must not panic the host.
   `clippy::{indexing_slicing, unwrap_used, expect_used, panic}` warn in
   non-test code.
3. **C++ interpreter compatibility.** Match guest-visible results, system
   calls, exception kinds, and retired-instruction counts in `--accurate`
   mode. The C++ `rvlinux` interpreter is the reference.
4. **Measured performance.** Target C++ tail-call dispatch performance and
   approach threaded dispatch, approximately 4% faster on CoreMark. Measure
   CoreMark, Dhrystone, and STREAM from `../libriscv/binaries`.
5. **Idiomatic embedding API.** Express the C++ architecture through Rust
   ownership, traits, and error handling.

## Scope

RV64 only: I, M, A, C, F, D, Zba, Zbb, Zbc, Zbs, Zicond, Zbkb,
Zfh(min), Zcb, Zimop/Zcmop, and V with Zvbb and Zvfhmin (RVA23U64).

The port covers ELF loading, arena memory with virtual paging, dynamic
executables through `ld.so`, Linux/newlib/minimal system calls, VM calls,
preemption, copy-on-write forks, serialization, threads, native heap/libc,
and debugging interfaces. The phase checklists distinguish completed and
planned features.

| Excluded C++ component | Files | Disposition |
|---|---|---|
| Binary translation to C | `tr_*.cpp`, `tr_*.hpp` | Replaced by safe Rust AOT translation |
| libtcc JIT | `tr_tcc.cpp` | Replaced by the Rust backends |
| asmjit JIT | `asmjit/` | Replaced by Cranelift |
| Assembly dispatch | `amd64/` | Requires assembly |
| Windows-specific shims | `win32/` | Deferred until Linux parity; `std` replaces most shims |
| RV32 | `rv32i.cpp`, `W == 4` paths | Removed on 2026-11-12; no width generics |
| RV128 | `rv128i*.cpp`, `rva128_instr.cpp` | Outside the supported architecture scope |
| C API | `../libriscv/c/` | Would require a separate crate permitting unsafe code |

## Architecture decisions

- **RV64 types.** Registers and addresses are `u64`; `Machine`, `Cpu`, and
  `Memory` are non-generic. The original RV32/RV64 `Width` trait was removed.
- **Errors.** `MachineError { kind, message, data }` replaces
  `MachineException`. `ExceptionKind` preserves C++ enum ordering.
  Dispatch returns `Result`; release builds use `panic = "abort"`.
- **Instructions.** `Instr(u32)` replaces `union rv32i_instruction`, with
  format accessors such as `instr.itype().signed_imm()`.
- **Dispatch.** Handlers tail-call each other. On stable Rust these are
  ordinary calls that LLVM turns into jumps; the opt-in `nightly-dispatch`
  feature uses guaranteed tail calls with `become`. See "Dispatch".
- **Features.** Behavioral CMake options map to Cargo features, including
  `ext-a`, `ext-c`, `ext-f`, `ext-v`, `fcsr`, and `debug-checks`.
  `ext-c` selects one decoder entry per 2 bytes instead of 4, affecting
  block structure and timeout locations. Default features match the
  default C++ decoder configuration. Dispatch-selection options have no
  equivalent; JIT options use the Rust backend.
- **Memory.** A lazily committed, zero-initialized arena has one padding
  page for wide accesses starting at its last valid address. Guest reads
  use `[0x1000, arena_size)` and writes use
  `[initial_rodata_end, arena_size)`. Other accesses use virtual paging,
  matching `RISCV_FLAT_RW_ARENA` plus `RISCV_VIRTUAL_PAGING`.
- **System calls and userdata.** Handlers are per-machine
  `fn(&mut Machine) -> Result<()>` entries. Userdata is an owned
  `Box<dyn Any + Send + Sync>`, checked by type through `set_userdata`,
  `userdata`, `userdata_mut`, and `take_userdata`. An `Arc` shares host
  state. Forks do not inherit userdata. Generic userdata was rejected
  because access occurs in host callbacks, outside dispatch.
- **Custom instructions.** `set_on_unimplemented_instruction` installs a
  per-machine hook inherited by forks. Only the decoder's final fallback
  invokes it, during execution rather than decoding, so shared segments
  remain independent of machine hooks. Custom printing awaits a disassembler.
- **Instruction definitions.** Each instruction implements `isa::Op`.
  Generic dispatch handlers (`seq`, `function`, `funcblock`) reuse the
  precise `exec`; block-ending instructions have dedicated tail-call handlers.
- **Host memory access.** Borrowed slices and typed helpers replace raw
  pointers. POD values use explicit little-endian conversions.
- **Floating point.** `f32`/`f64` operations use explicit NaN-boxing helpers.
  Full FCSR rounding and flag emulation remains unimplemented.
- **Threads and forks.** `Machine` is `Send + Sync`, checked at compile
  time. Forks share pages through `Arc`; execute segments use a process-wide
  shared cache.

## Toolchain

The library, the interpreter, the JIT and `rvlinux` build on stable Rust.
`rust-toolchain.toml` pins the stable release the repository is tested with
(1.98), so new compiler or clippy releases do not break builds or
`-D warnings` unannounced.

Two pins exist, for the opt-in parts that need a nightly:
`nightly-2026-09-28` (rustc `d080e7dff`), named in `fuzz/rust-toolchain.toml`
(cargo-fuzz needs nightly) and `NIGHTLY` in `.github/workflows/ci.yml`. Use
it with `cargo +nightly-2026-09-28` for the `nightly-dispatch` feature and
`aot`, which enables it. That dispatch requires `explicit_tail_calls`,
`rust_preserve_none_cc`, and `fn_align`. Explicit tail calls remain an
incomplete compiler feature; differential tests and instruction-count checks
detect regressions. AOT-generated code uses `become` too, and
`AotOutput::write_shared_object` compiles it with the toolchain libriscv was
built with.

The stable pin includes `riscv64gc-unknown-linux-gnu` for Rust test and
example guests built by `testutil::build_rust` and linked with RISC-V GCC;
running the tests with the nightly needs that target installed for it too.

Update either toolchain separately from unrelated changes. Run the full suite,
including `--ignored`, in both dispatch modes, repeat benchmarks, and record
baseline changes. `tail_call_tests.rs` verifies that indirect `become` calls
do not grow the stack in unoptimized builds; `tests/dispatch.rs`
(`handler_chains_do_not_grow_the_stack`) checks both modes, and with
`--release` that the stable dispatch's calls are jumps.

`.cargo/config.toml` aligns every function to 64 bytes on x86-64 and
AArch64 (`-C llvm-args=-align-all-functions=6`) for the stable dispatch,
which cannot align just its handlers (see "Dispatch"). It applies to builds
in this workspace only, and not when `RUSTFLAGS` is set: Cargo ignores the
`.cargo/config.toml` of dependencies, so a program that uses `libriscv` as a
crate gets the unaligned stable dispatch unless it passes the flag itself.
`-align-all-functions` is an LLVM option rather than a Rust flag, so a
toolchain bump can change or remove it; re-run the benchmarks.

## Dispatch

Dispatch uses **tail-call threading**, following C++ `tailcall_dispatch.cpp`
(Clang `musttail` and `preserve_none`). Each handler tail-calls the next
through a function pointer, giving each instruction its own indirect branch.
A `match` dispatcher shares one branch; the C++ switch dispatcher measured
approximately 28% slower than computed goto on CoreMark.

There are two modes with one set of handlers. `dispatch/abi.rs` is the only
code that differs between them: the handler signature and ABI, the
`Frame` a handler body sees, and the transfer macros (`execute!`,
`begin_block!`, `next_block!`, `unchecked_jump!`, `overflow_checked_jump!`,
`tail!`, `resume!`). Handlers in `dispatch.rs` and `bytecode.rs` are
written once with `handler!`/`cold_handler!`, read their arguments through
`Frame` methods, and transfer only through those macros.
`tests/dispatch.rs` (`nightly_dispatch_syntax_stays_in_abi`) rejects
`become`, `rust-preserve-none`, and `rustc_align` elsewhere.

- **`nightly-dispatch`** (opt-in, pinned nightly): `become`,
  `extern "rust-preserve-none"`, 64-byte aligned hot handlers, and every
  value in registers. The C++ mapping below describes this mode.
- **Default (stable)**: `return handler(...)` in tail position, with the
  Rust ABI. LLVM turns these into jumps (sibling calls) only when every
  argument is in a register, at most six on x86-64. So the signature is
  `fn(exec, i, m, ins, counter) -> Exit`: the limit lives in
  `DispatchState::limit` (read at limit checks, written after host calls),
  handlers load the decoder cache's address and length from `exec`, and
  block-ending handlers compute their own PC from `i` instead of receiving
  the block-end PC. `ins` is fourth, in rcx for variable shifts.

Stable-mode findings (2026-11-16, Ryzen 9 7950X, fixed-iteration CoreMark,
`perf stat`):

- Packing the index and instruction bits into one register, to also pass
  the cache in registers, put a load on the dependency chain from one index
  to the next: 1.9x the cycles of `nightly-dispatch`. Passing the cache
  and loading each handler's own bits instead ran 20% more host
  instructions for the same cycles as the chosen signature.
- A handler that passes a stack value's address to a call (any large
  `Result` returned through memory: system calls, vector, atomic and custom
  instructions through `O::exec`, `jit.run`) gets no jumps at all: LLVM
  leaves every later call a real call. Such host code runs in
  `abi::host` (a `move` closure; not inlined in this mode), and native code
  in `dispatch::native_step`. A disassembly scan of all handlers found no
  remaining indirect calls in tail position.
- Without per-function alignment (`fn_align` is nightly-only) the stable
  dispatch took 15-20% more cycles; aligning all functions
  (`.cargo/config.toml`) closed the gap. Five interleaved runs, median
  cycles for 4000 CoreMark iterations (rv64imafdc): HEAD before the split
  8.24G, `nightly-dispatch` 8.39G, stable 8.48G (+1-3%, within this
  machine's noise). Host instructions: 27.81G nightly, 29.05G stable
  (+4.5%); rv64imafdc_zba_zbb_zbs 27.84G / 30.77G.
- On AArch64 (96-core Neoverse-N1, stable 1.98, `rvlinux -n`, CoreMark
  rv64gc, median of seven interleaved runs, iterations/s) alignment closes
  most but not all of the gap: `nightly-dispatch` 1126, stable unaligned
  1011 (-10.2%), stable aligned 1084 (-3.7%). Alignment does not change
  `nightly-dispatch` (1125). It grows `rvlinux`'s text by 3.5% (2.8% with
  `nightly-dispatch`).

Sibling calls are an optimization, not a guarantee: unoptimized builds make
real calls (~500 bytes of stack each). The dispatcher therefore runs in
slices (`MachineOptions::dispatch_slice`, default `DISPATCH_SLICE`: 65536
instructions, 256 unoptimized, unbounded with `nightly-dispatch`). A slice
ends at the first limit check past its length and returns to
`dispatch_sliced`, which continues where it stopped; this is the background
JIT's slicing, which already had to be unobservable. A host call that ends a
slice (a new limit) begins another. Every loop passes a limit check, so a
chain is bounded by the slice plus the longest run of forward branches and
fall-throughs. `rvlinux --dispatch-slice 7` runs every difftest scenario a
second way (`<name>::interp_sliced`); results and counts must match.

C++ dispatch mapping:

| C++ (`tailcall_dispatch.cpp`) | Rust |
|---|---|
| `INSTRUCTION(bc, name)`: `static TcoRet<W> name(DecoderData<W>* d, DecodedExecuteSegment<W>* exec, CPU<W>& cpu, addr_t pc, InstrCounter& counter)` | `extern "rust-preserve-none" fn name(exec: &DecodedSegment, cache: &[DecoderEntry], i: usize, m: &mut Machine, counter: Counter, ins: u32, pc: u64) -> Exit` with one signature for all handlers. The order puts `ins` in rcx, where variable shifts want their count (cl) |
| `MUSTTAIL return computed_opcode<W>[bc](...)` | `become (entry.handler)(..., entry.instr)`. Store the handler `fn` in the decoder entry itself, so there is no second table lookup |
| `VIEW_INSTR_AS(fi, FasterItype)` reading `d->instr` | The caller passes the entry's (re-encoded) bits as `ins`, so a handler never looks up its own entry |
| `d += 1` pointer bump | index `i + 1`; read with `exec.cache.get(i)`. That is one bounds check per instruction (a compare and a predicted branch), the price of safe Rust |
| `__attribute__((preserve_none))` | `extern "rust-preserve-none" fn` (`feature(rust_preserve_none_cc)`). Without it every handler saved and restored six callee-saved registers: it cut CoreMark's host instructions by 17% |
| (handler placement left to the compiler) | Hot handlers are 64-byte aligned (`#[rustc_align(64)]`, `feature(fn_align)`). Otherwise the interpreter's speed shifted by a few percent with wherever unrelated changes placed them; aligned, fixed-workload CoreMark took 2.5–5% fewer cycles (32-byte alignment did not help) |
| `return RETURN_VALUES()`, `next_execute_segment(...)` exits | `become halt_exit(...)` etc.: exits are tail calls to cold handlers too, so a fast handler makes no calls and needs no stack frame |
| `BEGIN_BLOCK()` adds the block's instruction count; `OVERFLOW_CHECK()` only on branches/jumps | Same. The counting has to match exactly to keep instruction-count parity |
| `cpu.trigger_exception()` throws | Store the `MachineError` in the machine and return `Exit::Fault`. The return value is a one-byte enum |
| `next_execute_segment()` on out-of-segment jumps | Return `Exit::Jump`; `dispatch_simulate` selects the segment and re-enters with `BEGIN_BLOCK`. The chain must return to release its segment borrow. A jump to a block-start STOP in another usable segment counts the target block and stops without switching, avoiding two atomic reference-count updates per VM call (empty `vmcall`: 65 to 23 ns) |

Ownership: the decoded segment sits behind an `Arc` (C++ already shares
segments via `shared_ptr`). The dispatcher clones it out, so `&exec` and
`&mut machine` are disjoint borrows. `Counter` is the value/limit pair
passed in two registers.

**Semantics follow `cpu_dispatch.cpp`.** The C++ tail-call dispatcher differs
observably from its threaded/switch dispatcher (`cpu_dispatch.cpp`), and the
oracle is the default threaded build. Where they differ, the Rust port
follows `cpu_dispatch.cpp`:

- the counter is recorded before an illegal-opcode fault (tailcall leaves it);
- an ECALL whose handler stops the machine or changes PC resumes at
  `regs.pc + 4` (tailcall resumes at `regs.pc` when it is outside the segment);
- other SYSTEM instructions retrieve both counters, and if the machine stopped
  or PC changed, continue *at* `regs.pc` (tailcall continues at `pc + 4` when
  stopping);
- the entry check is `begin <= pc < end` (tailcall: `pc + 2 <= end`).

Handler chains have no fixed length. They end on counter overflow, `stop()`,
a system call requiring a dispatcher exit, or a fault (or a slice end).
Transfer between handlers only through the `abi` macros, which use `become`
with `nightly-dispatch`, to prevent stack growth. Phase 1 introduced this
dispatcher and its decoder cache; Phase 4 added specialized bytecodes.

### Bytecodes

The fast path ports C++'s threaded bytecodes:

- `decode::fast_handler` is `decode_bytecodes.cpp` + `threaded_rewriter.cpp`:
  it picks the bytecode and re-encodes the operands into the
  `threaded_bytecodes.hpp` formats (`FasterItype`, `FasterOpType`, ...,
  in `bytecode.rs`), doing every check that can be done at decode time:
  `rd != x0`, branch and jump targets inside the segment and aligned, shift
  amounts in range, forward branches (no counter check).
- `bytecode.rs` is `bytecode_impl.cpp`: one handler per bytecode, reading
  the pre-decoded operands. They never return a `Result`: the only fallible
  ones, loads and stores, read the flat arena inline (`Memory::arena_read`,
  which cannot fail) and tail-call a cold `*_slow` handler for the page
  table and faults. C++ gets the same effect from exceptions. In between,
  a `*_cached` handler serves page cache hits outside the arena
  (`Memory::read_cached`/`write_cached`, stores only to unshared pages of
  their own) without calls, so it needs no stack frame: every load and
  store of a machine without an arena (forks) goes there.
- Slow paths (`RV32I_BC_FUNCTION`, `RV32C_BC_FUNCTION`, `FUNCBLOCK`) keep
  the original instruction and run the precise handler, as in C++.

Where C++ sends an instruction to a FUNCTION slow path that behaves exactly
like a bytecode, this port uses the bytecode (results, counts and faults are
the same; the rewriter's C++-exact bits are kept for fault data, see
`decode::cpp_entry_bits`):

- C.MUL: `c_op::<Mul>`, like C.SUB.

Both implementations now select C.LD (`RV32C_BC_LDD`) and SRAIW
(`RV64I_BC_SRAIW`) bytecodes. The upstream C.LD operand swap is fixed.
Previously, the C.LD slow path consumed about 10% of CoreMark interpreter time.
Upstream `21d86a87` added C.SUB, C.AND, C.SUBW, C.ADDW, C.SRAI and C.LUI
bytecodes, which this port already had (`c_op`, `c_srai`, `c_lui`), and
C.FLD, C.FLDSP, C.FSD and C.FSDSP (`RV32C_BC_FLD`/`FSD`, here `c_fld`/`c_fsd`).
`e5fe5d27` added SH1ADD.UW and SH2ADD.UW (`RV64I_BC_OP_SH1ADD_UW`/`SH2ADD_UW`).
SH3ADD.UW stays on the slow path, as in C++.

`rvi_fuse.cpp` is not ported: nothing in C++ calls `try_fuse`.

### Fused pairs

Upstream `a94a8c14` fused 128 common pairs of adjacent instructions
(`fused_pairs.inc`), and `fused.rs` ports them, the pair list unchanged. After
a segment is decoded and its blocks measured, an entry whose next
instruction is in the same block, and whose bytecode forms a pair with the
next one, gets the pair's handler, `fused::<(A, B)>`. That handler runs A's
body and continues in B's handler with a direct jump (`execute_to!`, C++
`NEXT_INSTR_TO`) instead of the indirect one through B's entry. B's entry is
unchanged, as it may be a jump target. Counts are added on block entry, so
they do not change.

- The bodies are shared through marker types in `bytecode.rs` (C++
  `bytecode_bodies.hpp`): `First::run` is a handler's body without its
  transfer, and `Bytecode::HANDLER` its handler. The `sequential!`,
  `loads!` and `stores!` macros expand each body into both. A load or store
  outside the arena continues in its own `*_cached` handler, which then
  dispatches B through its entry as usual.
- Bytecodes are not identified by handler address: a generic handler
  (`op::<Add>`) can be instantiated more than once, and LLVM merges
  identical ones (`fast_jal`, `c_jmp`). `decode::fast_handler` tags the
  entries it picks a fusable bytecode for with a `fused::Bc`, and fusing
  goes by those tags. Entries the block measurement forces to end a block
  (`force_invalid`, `force_funcblock`) lose their tags.
- C++ `unfuse_entry_before`: `create_block_ending_entry_at` (breakpoints, the
  libc fast path) first decodes the entry before the patched one again,
  which gives it its own handler back. These patches are only applied
  while a segment is created, before native code is installed. Installing
  JIT or AOT entry points does not unfuse, unlike C++'s livepatch `claim`:
  a pair before a native entry runs B in the interpreter, with the same
  result, and enters native code at the next block.

CoreMark, 3000 iterations, `rvlinux -n`, Ryzen 9 7950X, stable dispatch,
median of five interleaved runs (`perf stat`): rv64imafdc 6.80G to 5.99G
cycles (-11.9%), rv64imafdc_zba_zbb_zbs 6.07G to 5.58G (-8.2%). Host
instructions barely change (-0.3%, -0.1%); the direct jumps predict better.

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
complete process. Freestanding builds use `crates/testutil/guests/coremark`,
with sources from `../coremark` or `$COREMARK_DIR` (eembc/coremark `eefc986e`),
4000 iterations, `-O2`, and the best of five process wall times, measured on
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
rule, so variants that do not count exactly had to be validated by their
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

Compilation dominates this workload: the 460 KB libc segment takes about
1.9 s in Cranelift before CoreMark executes in 260 ms; asmjit compilation
takes about 150 ms. Lazy compilation is planned in Phase 4. Subsequent
parallel compilation on up to 16 threads reduced segment compilation from
3.1 to 0.29 s on a 28-core Xeon Gold 6132, with 18% more CPU time and
identical output.

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

The prebuilt RV64 ELFs in `../coremark` are built the same way (passed
with absolute paths: the bench runs in `crates/rvlinux`). Measured at
`0d31d48` on 2026-11-13 against upstream `e806d460`, on an otherwise idle
machine. CoreMark's own Iterations/Sec, one bench run; the first two rows
are the bench's own builds:

| ELF | Rust interpreter | C++ threaded | C++ tailcall | Rust JIT | C++ asmjit |
|---|---:|---:|---:|---:|---:|
| rv64imafdc | 2425 | 2440 | 2197 | 19076 | 19326 |
| rv64imafdc_zba_zbb_zbs | 2777 | 2442 | 2290 | 19778 | 19640 |
| coremark-rv64g | 2492 | 2680 | 2380 | 18412 | 18836 |
| coremark-rv64gc | 2416 | 2330 | 2255 | 18700 | 19479 |
| coremark-rv64g_b | 2744 | 2716 | 2618 | 20839 | 20095 |
| coremark.rv64gb.gcc14.2.elf | 2809 | 2863 | 2596 | 17893 | 19131 |
| coremark-rv64g_zba_zbb | 2655 | 2779 | 2496 | 19559 | 19445 |

Two more interleaved rounds under `perf stat`, as the Rust variant's speed
relative to C++ by cycles per iteration (corrected as described below):

| ELF | Rust interpreter vs C++ threaded | Rust JIT vs C++ asmjit |
|---|---:|---:|
| coremark-rv64g | −8.6% | −2.2% |
| coremark-rv64gc | +2.2% | −3.2% |
| coremark-rv64g_b | −2.1% | +3.3% |
| coremark.rv64gb.gcc14.2.elf | −3.9% | −5.6% |
| coremark-rv64g_zba_zbb | −2.7% | 0.0% |

The Rust interpreter is within 2-4% of C++ threaded on most ELFs, ahead on
rv64gc and on the Zbs build (+14%), 9% behind on rv64g, and ahead of C++
tailcall everywhere. It runs ~40% more host instructions per iteration
(6.5-7.0M against 4.6-5.0M) at a higher IPC. The Rust JIT is within −6% to
+3% of asmjit: level or ahead on rv64g_b, rv64g_zba_zbb and the Zbs build,
furthest behind on the gcc14.2 ELF, where it runs more host instructions
than asmjit. Compiling up front costs the Rust JIT ~8x asmjit's time:
250-340 ms against 30-40 ms on most ELFs, ~1.4-1.5 s against ~170-185 ms
on rv64gc and rv64g_zba_zbb (outside CoreMark's timed region). Compiled in
the background, the JIT can arrive after CoreMark has calibrated on the
interpreter, which then picks too few iterations and fails the 10-second
rule.

Measuring with `perf stat`: with `ITERATIONS=0` CoreMark first runs batches
of 10, 100, … iterations until one batch N takes ≥ 1 s, then the reported
N·(1 + 10/⌊secs⌋). Divide cycles by the reported count plus
10 + 100 + … + N (11110 more for the interpreters, 111110 for the JITs),
not by the reported count alone: that overstates by 25-37%, by a different
amount when two variants land on different counts (rv64g: Rust 30000,
C++ 40000). For the JITs also subtract the up-front compile, measured by
the same command with `--fuel 1`. With both, cycle ratios matched the score
ratios to within 0.2% on every ELF. The remaining JIT difference is mostly
register allocation:
guest registers live across a whole region (loaded at entry, stored at
every exit, as in asmjit), and in high-pressure loops regalloc2 keeps
loop-carried ones in spill slots. (Guest calls were still tail calls
with returns through the entry table then; see "Native calls" under JIT.)

## JIT

The default `jit` feature provides the Cranelift backend, replacing C++ asmjit.
`MachineOptions::jit` enables it per machine. `rvlinux` enables JIT execution
by default; `-n`/`--no-translate` or `--no-jit` selects interpretation.
`--no-default-features` excludes the runtime and Cranelift. Interpreter-only
`libriscv` builds must explicitly select ISA features; `rvlinux` selects them.

Instruction selection and region analysis are in `codegen/`; compilation and
caching are in `jit/`. Executable memory and native entry are isolated in
`libriscv-jit-rt`. JIT execution supports x86-64 and AArch64
(`emit::HOST_SUPPORTED`); other hosts interpret. On AArch64,
`CodeMemory::new` cleans the data cache and invalidates the instruction cache
for new code (`dc cvau`/`ic ivau`, as `__clear_cache`), and `CodeMemory::enter`
issues an `isb`, because background compilation writes code on another thread.
Cranelift's AArch64 backend rounds and fuses natively, so its code has no
libcalls. Cranelift x64 tail calls require `preserve_frame_pointers`.

### Unsafe boundary

`libriscv` generates and links code through safe APIs and enforces
`forbid(unsafe_code)`. `libriscv-jit-rt` maps executable memory with `memmap2`,
loads cached objects, validates entry tables and memory layouts, and calls a
trampoline. Its `unsafe_code = "deny"` policy requires documented exceptions.

Generated code must enforce arena bounds or address masking. External calls
are limited to region functions, floating-point rounding/FMA helpers on hosts
without SSE4.1/FMA, and the host dyncall trampoline. Code is position-independent:
JIT libcalls use an appended GOT pointing to Rust helpers; saved objects use
the system linker's GOT and `libm`.

### Regions and dispatch

Each region is a Cranelift `tail` function:
`(state, pc, counter, depth) -> (target, counter)`. Regions follow fall-through,
direct branches, and call return addresses. They stop at called-function
entries, indirect jumps, unsupported instructions, or other regions. The normal
limit is 512 instructions, extended to 1024 to retain loops of at most 1 KiB
whose headers are already included. Keeping STREAM's approximately
900-instruction `-O0` loop intact increased Copy throughput from 2.1 to 17.9 GB/s.

Region transfers use native tail calls, directly or through an entry table.
Guest registers are SSA variables loaded at entry and stored on exit; FP
registers retain raw bits and x0 remains zero. Instruction selection follows
`decode::fast_handler`, including compatibility quirks. Unsupported instructions
return to the interpreter with matching PC, registers, and counter.

Only decoder block starts are patched with `dispatch::jit_enter`. Mid-block
native entries, including loop headers, are selected by `dispatch_chain` when
execution begins a block there. Patching fall-through entries would double-count
instructions. FUNCBLOCK entries compile the precise instruction followed by the
usual block-end check, except encodings rejected by the precise handler but
accepted by the fast path, such as FMADD.H/.Q.

Block entry adds the decoder count; checks occur at interpreter-equivalent
block ends. Mid-block exits subtract the target's count before continuation.
Stale segments are interpreted. Native instructions cannot make a segment stale;
host calls recheck staleness.

### Floating point

F/D compilation is disabled with `debug-checks`, which enables NaN-boxing.
Arithmetic otherwise uses host IEEE round-to-nearest operations, matching the
interpreter's default non-FCSR behavior. NaN payloads depend on the host.
FMIN/FMAX return the numeric operand for one NaN and canonical NaN for two.
Integer FCVT honors the encoded rounding mode, including DYN from FCSR.frm,
saturates, and converts NaN to the maximum. Single-precision writes preserve
the upper register half without NaN-boxing.

**FNMADD** (and the vector VFNMADD/VFNMACC) computes `-(rs1*rs2) - rs3` as
`fma(-rs1, rs2, -rs3)`. That is the spec's definition, and it differs from
`-fma(rs1, rs2, rs3)` only in the sign of an exact zero. An example is
rs1 = +0, rs2 = -0, rs3 = +0: the spec gives +0, the negated fma gives -0.
The interpreter, the JIT and AOT all use the same formula.

Not reproduced (C++ bug): C++ uses `-fma` (`rvf_instr.cpp`, and
`rvv_instr.cpp` for the vector forms), so on x86-64 it returns -0. On
AArch64, where FMA is baseline, an optimizing compiler folds `-fma` into the
host's FNMADD instruction, which gives +0. The C++ result therefore depends
on the host and the build: the reference emulator built with `gcc -O2` on
AArch64 returns +0. The port used to copy `-fma`, with the same effect. On
AArch64, its release interpreter and AOT code returned +0, while the debug
interpreter and the Cranelift JIT returned -0; the AOT test
`fp_special_values_match_up_to_nan` caught it.

NaN results still aren't canonicalized, so their sign and payload can differ
from C++ as before.

### Dyncalls

Valid `dyncall.rs` words invoke the system call selected by their immediate.
They cannot change PC and normally execute mid-block. `dispatch::run_dyncall`
publishes PC, counter, and limit. A stop or penalty reaching the retired count's
limit halts immediately after the call; a limit crossed only by the block's
count is checked at the block end. Stale segments are left. Forced block ends
retain the dyncall bytecode. Interpreter and JIT results, faults, and counts
match when handlers obey the encoded register contract.

Mid-block JIT dyncalls use `JitState::host_call`. Native code stores promised
integer inputs and outputs, FP registers unless excluded by the encoding, and
`sp` for nested preemption. Outputs are stored before the call so an unwritten
output retains its current value. Afterwards it reloads the counter and outputs.
Region summaries include these registers for native callers.

The host trampoline invokes the per-machine handler and reacquires the register
files and arena. Stops, faults, limit-reaching penalties, and stale segments
return `EXIT_HOST`; block-ending dyncalls use `EXIT_DYNCALL`. A moved arena
forces an exit because callers retain its pointer. Moved register files abort
the process, violating the borrowed-host contract. Handler panics are caught
and resumed after native code returns. Handler lookup remains dynamic because
tables are per machine and cached code contains no absolute host addresses.

Measured on 2026-11-15: a `dyncall; ret` loop improved from 13.8 to 6.7 ns/call
(interpreter: 9.9 ns), and from 320 to 179 host instructions/call. Asteroids
improved from 2.31 to 1.74 µs/frame over 4,770 frames and 20 seeds
(interpreter: 3.8 µs/frame).

### Bounds checks and register pressure

Arena windows are constants in compiled code and part of `MemoryLayout` and
shared-segment keys. `CodeMemory::enter` requires them to fit the machine's
windows and arena. The loader finalizes windows before creating segments.
Checks lower to `lea`/`cmp`/`jae`; the instruction limit is read from state at
each check. Exit arguments are materialized in separate cold blocks.

One check can cover the offset span of accesses through an unchanged base
register within a straight-line run. Labels, control transfers, and FUNCBLOCK
end a run. A store check can cover enclosed loads when the write window lies
inside the read window. Span checks preserve individual start-address checks;
failure exits at the first access for interpretation. The
`merged_bounds_checks_match_the_interpreter` test sweeps window boundaries.

Backward conditional branches keep the checked loop path inline, matching
asmjit's layout. This reduced a five-instruction loop from approximately three
to two host cycles/iteration and CoreMark rv64gc time by about 4% on 2026-11-13.
Removing fall-through region entries increased instructions by 2% and branch
misses by 12%, so that optimization was rejected.

### Native calls

Guest calls whose return address is a label in the caller's region use native
calls, including indirect calls through the entry table. The caller stores
registers the callee's `Summary` may read and reloads those it may write.
Matching returns continue at the return label; other targets propagate to the
caller. Guest returns through `ra` or `t0` return a target. Host exits return
`DONE`, propagated through all callers. The trampoline dispatches other targets
through the entry table or returns `EXIT_INDIRECT` when no entry exists.
Beyond `DEPTH` (32), calls use tail calls to bound host stack use.

On 2026-11-14, freestanding CoreMark (30,000 rv64imafdc iterations) improved
from 1.28 to 1.07 s with N-bit accesses, 1.57 to 1.35 s with checked accesses,
and 1.61 to 1.38 s with `--accurate`. Static glibc compilation increased from
1.5 to 2.0 s, mainly in regalloc2 around calls that clobber Cranelift's `tail`
register set. Raising region limits to 2048 instructions increased compilation
time by another 15% without improving execution.

### Background compilation

`MachineOptions::jit_background` is the `rvlinux` default. A worker compiles an
`Arc<DecodedSegment>` and publishes a patched copy through `OnceLock`
(`finish_background_jit`). Every machine sharing the segment receives the
result, including forks. Safe Rust avoids mutating handlers in a running cache.

While compilation or installation is pending, dispatch runs in
`jit_background_slice` slices, default 2^20 instructions. Between slices,
finished copies replace the machine's segments. Compilations started during an
unsliced run begin slicing at the next limit check returned by host code or a
segment change (`begin_slice`). Deferred rebuilds also use slices to track heat.

Slice boundaries preserve block counting, real limits visible to handlers, and
observable PC updates. Stops and handler changes to the real limit end slicing.
CSR instructions that halt at their own PC use the next block for slice-only
limits to avoid executing the CSR twice. `jit::tests` checks these cases with a
compilation that never finishes.

Stale results are discarded; original and compiled copies share a stale flag.
`Machine::wait_for_jit` waits for all compilations held by the machine, including
ones started elsewhere. A panic publishes an empty result to release waiters.

### Code cache

`jit/cache.rs` saves Cranelift output for reuse. `JitCacheOutput` records each
compiled `emit::Image`; saving machines compile their own segments instead of
using cached or shared segments. `write_object` writes an ELF relocatable
object without additional dependencies. `write_shared_object` links with
`$CC` or `cc -shared -nostdlib ... -lm`.

Each `libriscv_jitcache` record contains magic, PC-relative code and metadata
offsets (`R_X86_64_PC64`, `R_AARCH64_PREL64`), and lengths. Libcalls (x86-64
only) use `R_X86_64_GOTPCREL`.
The weak `libriscv_jitcache_table` symbol supports `dlsym`; embedded objects
use `__start_`/`__stop_` symbols and `SHF_GNU_RETAIN`. An empty runtime record
ensures section symbols exist. Shared objects need no runtime relocations in
these read-only sections.

`jit_cache::load_shared_object` (`--jit-load`) registers shared objects.
`RVLINUX_JIT_EMBED=a.o:b.o` embeds objects at build time. Both use a process-wide
registry consulted when creating segments if `jit_cache` is enabled, including
when per-machine JIT compilation is disabled. `CodeMemory::linked` retains the
shared object while its code is used.

`codegen::NativeKey` includes segment address, length, CRC32-C, a 64-bit code
hash, execute-only and libc patch settings, memory layout, and a build
fingerprint. The fingerprint covers version, host ISA, Cranelift flags,
extension features, `JitState` offsets, JIT sources, and `codegen::SHARED_SOURCES`.
Mismatched builds, CPUs, and layouts are ignored; `verbose_loader` reports
lookups. CRC32-C and FNV are forgeable, so metadata `RVJITM02` also stores the
original guest bytes and `Meta::fits` requires an exact match. `RVJITM01` units
are ignored. Colliding units remain distinct. Breakpoint segments are excluded.
Loading or linking native objects executes trusted host code; byte matching
does not validate malicious native code.

Loading a static glibc C++ program's 20 MB image took 0.14 s versus 12 s to
compile in a debug build. Tests cover library and CLI APIs, multiple segments,
libcalls through `libm`, invalid files, dynamic executables, and embedded
objects. Slow dynamic and embedded tests are ignored by default. Re-exporting
cached code remains unimplemented.

### Cache directory

`MachineOptions::jit_cache_dir` uses a `CacheDir` with a path, initial-segment
policy, and address ranges. Files use
`<dir>/<build fingerprint>/<vaddr>-<key hash>.so` and are loaded once per process.
Writes link in a private, unpredictable mode-0700 temporary directory, create
objects with `create_new`, and rename the result atomically.

Saving requires code verified against a host-selected file: an explicitly
covered initial segment, or a pinned-file mapping with exact bytes re-read by
`Memory::came_from_pinned_file`. The latest 64 mappings are tracked. Rebuilt
segments are never saved. Lookups require matching bytes as well as range.
Limits are 32 saves and 512 MiB per process, and 256 entries per build directory.
Directories and files must be owned by the effective user and not writable by
group or others; build directories and files must not be symlinks. These checks
prevent guest-controlled cache substitution and unsafe path access.

Background saving is part of compilation. Call `jit_cache::wait_for_saves`
before exit; worker threads do not keep the process alive. Directory caching
preserves segment sharing and retains old build directories.

For dynamic executables, `rvlinux` caches its fixed dynamic linker and pinned
library slots, excluding the program. The path is `--jit-cache-dir`,
`$XDG_CACHE_HOME/rvlinux/jit`, or `~/.cache/rvlinux/jit`. `--no-jit-cache` disables
it. Exit waits for pending saves. With cache and dispatch fixes, `stream.dyn`
improved from 3.5 s wall time and 7.5 s CPU time to 0.88 s; compiled `libc.so`
occupies about 36 MB. Tests cover forged keys, permissions, temporary-file
creation, save limits, pinned-file provenance, and CLI directory reuse.

### N-bit address space

`jit_automatic_nbit_address_space` (`--nbit-address-space`) uses
`arena + (addr & (arena_size - 1))`. Native accesses can modify read-only data
and page zero and wrap outside addresses into the arena, bypassing paging.
Interpreter semantics are unchanged. Arena size rounds up to a power of two;
4000 MiB becomes 4 GiB. Rounding by more than 256 MiB returns `OutOfMemory`.
The arena is enlarged with the mask, avoiding the C++ translator's possible
out-of-mapping accesses.

For 4 GiB, the JIT emits a 32-bit add and zero extension; LLVM derives this
from AOT masking. Offsets are added before truncation. `CodeMemory` verifies
that the mask plus the widest access fits the padded arena.
`tests/nbit_arena.rs` and JIT `nbit_*` tests cover these rules.

### Instruction limits

`jit_ignore_instruction_limit`, enabled by `rvlinux` unless `--accurate`,
disables native counting and limit checks. Native loops run until leaving
native code. Interpreter blocks still count and enforce the limit; system
calls still allow `stop()`. Counters therefore undercount total execution.
The setting is part of `MemoryLayout` and all native-code keys.
`native_enter` retains entry counts rather than subtracting them, including
after dyncalls, so counters remain monotonic.

`rdinstret` observes the undercount; instruction-clock CoreMark can fail its
10-second rule despite valid CRCs. Removing native counting reduced
freestanding CoreMark (30,000 iterations, N-bit) from 1.21 to 1.10 s for AOT
and 1.48 to 1.30 s for JIT. `unlimited_*` tests cover both backends.

### Speculative execution

N-bit masking constrains each access to an arena of at least `mask + 8` bytes.
Checked JIT code uses `select_spectre_guard` conditional moves on bounds checks
and entry-table lookups. Guards are enabled by default through
`jit_spectre_guards`; `--no-spectre-guards` disables them for trusted guests.
The setting is part of `MemoryLayout`, preventing incompatible cache reuse.
Guards cost about 25% of checked CoreMark native performance. AOT and interpreter
checked accesses currently rely on branches and do not provide these guards.

### Validation

Differential tests run interpreter, background JIT, and upfront JIT
configurations against C++. `testutil::run_rust` defaults to `--no-translate`.
`tests/jit.rs` compares complete machine state across handwritten and random
programs at approximately 50 limits, including background compilation, native
calls, and N-bit layouts. `LIBRISCV_JIT_FUZZ=<programs>[:<seed>]` extends runs.
FP comparisons are exact until the first NaN, then accept any NaN once;
bitwise FP operations and integer programs compare exactly.

Historical performance, 2026-11-12: freestanding CoreMark, 4,000 iterations,
`--accurate`, best of three. Later measurements are in "Baseline".

| | rv64imc | rv64imc_zba_zbb_zbs |
|---|---|---|
| Rust interpreter | 2081 ms | 1866 ms |
| Rust JIT | 252 ms | 246 ms |
| C++ threaded | 1876 ms | 1792 ms |
| C++ asmjit | 239 ms | 244 ms |

JIT execution was within approximately 5% of asmjit with exact instruction
counts. Compilation took about 60 ms of a 0.34 s process runtime, compared
with asmjit's 0.26 s total. Background compilation later saved 5–10 ms.

## AOT

The optional `aot` feature translates execute segments to safe Rust, replacing
C++ translation to C (`tr_emit.cpp`, `tr_translate.cpp`, `tr_compiler.cpp`).
It adds no dependencies and requires no unsafe code for embedded translations.
`aot/emit.rs` generates region functions, `structure.rs` lays out control flow,
and `mod.rs` manages keys, registration, and output.

### Usage and registration

`MachineOptions::aot_output` records translations of every created segment
while execution continues normally. `AotOutput::source()` produces one Rust
file defining `mod translation`, `META`, and `enter`, using only standard types
across the host interface.

- **Embedded:** include the source and call
  `aot::register(translation::META, translation::enter)`. The host crate needs
  `explicit_tail_calls` and `rust_preserve_none_cc`. `RVLINUX_AOT_EMBED`
  embeds listed files in `rvlinux`; the `aot` feature alone permits a build
  with no unsafe code in any crate.
- **Shared object:** `AotOutput::write_shared_object` uses `$RUSTC` or `rustc`
  with the build's toolchain unless `$RUSTUP_TOOLCHAIN` overrides it.
  Compilation uses `-C opt-level=3 -C target-cpu=native`; additional flags
  come from `$LIBRISCV_AOT_RUSTFLAGS`. Two C exports,
  `libriscv_aot_meta` and `libriscv_aot_enter`, form the unsafe boundary.
  `aot::load_shared_object` requires `aot-dylib` and uses the runtime's
  `dlopen` wrapper and borrowed `RawCall`. Translated code remains safe Rust.

Translations use `codegen::NativeKey`: segment address, length, CRC32-C,
64-bit code hash, execute-only and libc patch settings, memory layout, and a
fingerprint of AOT/shared sources and extension features. `MachineOptions::aot`
is enabled by default; matching translations take precedence over JIT.
`verbose_loader` reports hits and misses. Translating machines create their
own segments. Untranslated segments can be shared with machines that have AOT
disabled. Translation output includes the N-bit layout even if AOT execution
is disabled.

Generated source is independent of the host CPU. Shared objects compiled with
`target-cpu=native` record required x86-64 features in `META` (ABI version 2)
and are rejected on incompatible CPUs.

### Semantics and code generation

AOT shares instruction selection, regions, counting, and exit handling with
the JIT through `codegen` and `native_code`. F/D arithmetic uses host IEEE
operations. FMIN/FMAX, integer FCVT, and FCLASS follow interpreter behavior;
FCVT.S.D/D.S canonicalize NaNs. Other NaN payloads depend on the host.

Each region is an `extern "rust-preserve-none"` function:
`r<N>(state, integer_regs, fp_regs, arena, pc, counter, depth) -> (target, counter)`.
Modules contain 40 regions to allow parallel rustc code generation.
Guest registers are local values, loaded at entry and written at exit.
Transfers use direct `become`, an entry-table `become`, a guest return target,
or an `A` host-exit value. The entry table is initialized through `OnceLock`.

Arena parameters use `&mut [u8; N]`, sized for the layout's windows plus the
widest access. Start-address checks permit LLVM to prove indices in bounds
without extra slice checks. Unchanged base registers share offset-span checks
within straight-line runs. Region-wide checks for bases never modified by the
region or its native callees can replace repeated run checks. A wider failed
check may cause an earlier interpreter fallback without changing results.
N-bit accesses use masking instead.

Instruction limit and rounding mode are locals. Register files and arena use
separate mutable parameters so LLVM can infer `noalias`, preserve register
values across guest stores, and forward inlined register accesses.

### Native calls

Calls use native Rust calls when their return address belongs to the caller's
region. Regions retain return labels unless another function calls that label,
and normally contain at most 512 instructions. Direct targets include
`auipc`+`jalr` pairs recognized by `Analysis::jalr_target`. Such a `jalr` cannot
be a region entry or be separated from its constant-producing instruction.

The caller stores registers the callee may read and reloads those it may write.
`codegen::calls::Summary` propagates register effects to a fixed point; indirect
jumps conservatively include all registers. Matching returns continue in the
caller; other targets and `DONE` propagate outward. Beyond `DEPTH` (32), calls
become tail calls to bound host stack use.

### Inlining and devirtualization

Self-contained regions, which exit only through guest returns, native calls,
or host exits, qualify for `#[inline(always)]`. Including inlined callees,
size must be at most 256 instructions and size × call sites at most 1024,
unless size is at most 16. Recursive cycles are excluded. Larger limits
(512/2048) reduced CoreMark instruction counts but increased cycles.

Inlining folds constant-entry dispatch and register synchronization. Regions
with tail transfers are excluded because LLVM can convert an inlined `become`
to a regular call. Inlined regions exit with `EXIT_JUMP` past the depth limit
instead of tail-calling, preventing unbounded host stack growth.

When at most four labels have their addresses taken, indirect calls compare
the target against them and use direct calls and callee summaries on matches.
Address detection recognizes `auipc`, `lui`, or `li` plus matching `addi`.
An isolated `auipc` before a call is excluded. This optimizes CoreMark's
indirect comparator calls.

### Control flow

`aot/structure.rs` follows Ramsey's "Beyond Relooper" (ICFP 2022). Forward
branches become labeled `break`; back edges become loop `continue`. A root
`match pc` dispatches entries. Side entries and irreducible edges use copies of
loop bodies without their headers, completing a partial iteration before
joining the loop header. Nested loops are handled recursively.

If all entries would exceed 2× code growth, weak entries inside loops are
removed. The interpreter continues to the next available entry, at most the
loop header. Growth beyond 8× or unreduced control flow falls back to a
`loop { match pc { .. } }` dispatcher. LLVM does not thread constant PCs
through that loop header; this fallback measured 2.4× slower than JIT on a
static glibc program. Static glibc required one fallback region; CoreMark none.

**Performance** (2026-11-13, idle box, release `rvlinux`): static glibc
collatz 229 ms vs JIT 237 ms (interpreter 1893 ms); CoreMark 1.0 (bench
column "Rust AOT", `cargo bench -p rvlinux --features aot-dylib,jit
--bench coremark`) 18959 / 18141 vs JIT 19021 / 19734 and asmjit 19235 /
19670 (rv64imafdc / +zba_zbb_zbs), before span-merged checks.
2026-11-13 later, freestanding CoreMark (`guests/coremark`, 30000
iterations, wall time, rv64imafdc / +zba_zbb_zbs): JIT 1.64 / 1.60 s,
AOT 1.48 / 1.48 s (1.50 / 1.50 s before region checks and locals), with
`--nbit-address-space` (now 4 GiB at the default `-m`) JIT 1.45 / 1.40 s
and AOT 1.19 / 1.23 s. Native x86-64 `-O2` CoreMark runs the same
iterations in ~0.63 s. Checked builds retain one comparison per access in
pointer-chasing loops, where bases cannot be shared, plus precise exit handling.
2026-11-14, same CoreMark, rv64imafdc, N-bit: no instruction limit 1.10 s
(from 1.21), native calls 0.97 s, the call depth in a register and one
borrow per register file 0.89 s, no side entries in large regions
0.88 s (checked: 1.47 to ~1.2 s). Per iteration that is ~600k host
instructions and ~165k cycles against native's ~400k and ~136k; most of
the remaining gap is the register stores and reloads at calls and region
entries, and callers' spills around calls (`rust-preserve-none` saves
nothing). Static glibc translates (~470 KB of code) to ~19 MB of Rust in
0.1 s; `rustc` takes ~47 s for it on 32 cores (one module took 72 s,
single-threaded), CoreMark freestanding a few seconds.
2026-11-14 later, the same CoreMark, `perf stat` host instructions and
cycles per iteration, interleaved against a build of the previous commit:

| | instructions | cycles |
|---|---:|---:|
| N-bit, before | 597k | 168.5k |
| + inlining small leaves | 524k | ~158k |
| + devirtualization | 515k | 153.4k |
| + inlining every self-contained region | 506k | 151.2k |
| + `noalias` register files and arena | 457k | 134.6k |
| checked, before / after | 877k / 753k | 218.5k / 193.3k |
| native x86-64 `-O2` (gcc 13) | 401k | 135.8k |

N-bit AOT runs CoreMark at native speed (per function, every part was
within ~1% of native but the list sort's comparator calls, before the
`noalias` change). Separate borrows enabled inlining benefits by proving
that guest stack stores cannot modify register files. Previously, a 16-byte
register reload following separate stores missed store forwarding and
consumed 40% of `cmp_complex` cycles. The C calling
convention for regions was slower than `rust-preserve-none` (611k
instructions), and turning off LLVM's SLP vectorizer slower too. Static
glibc: `rustc` 44 to 47 s wall time (CPU 134 to 180 s), the shared object
14.0 to 16.4 MB.

**Testing.** `libriscv`'s `tests/aot.rs` (feature `aot-dylib`, needs
`rustc`): the JIT tests' programs (hand-written, 150 random integer and
150 random FP programs, FP special values, dyncalls, merged bounds checks
at the window edges with and without a program, layouts), each test
translating its programs into one shared object and comparing complete
machine state with the interpreter at many instruction limits;
`LIBRISCV_AOT_FUZZ=<programs>[:<seed>]` for longer sessions (3000 integer
and 2000 FP programs clean), and native calls
(`support::native_call_programs`, shared with `tests/jit.rs`: recursion
past `DEPTH`, also leaving for the host at the bottom, `auipc`+`jalr` and
indirect calls, a callee moving the caller's base register, a callee
leaving for the host, a return elsewhere, an inlined callee calling past
`DEPTH`, devirtualized indirect calls and one to another target, FP
registers across calls). `rvlinux`'s `tests/aot.rs`: freestanding
CoreMark through `--aot-save-shared`/`--aot-load`, layouts, `--no-aot`,
bad files; ignored because slow: static glibc (~1 min) and the embedded
translation (builds `rvlinux` again in `target/aot-embed`).

Not yet: lazy translation, translating only hot code, saving
translations of segments that came from the cache. For the calls that
are not inlined (large callees, tail-calling ones, indirect calls to
other targets): registers passed as arguments across region transfers
and calls instead of through the register file, and liveness-based
stores before calls (a callee's `Summary` counts every register it uses
as read, also those it writes first). Inlining removed most of what
these were for on CoreMark.

## Virtual paging

`memory/paging.rs` and `memory/page.rs` port paging support from `memory.cpp`,
`memory_rw.cpp`, `memory_helpers_paging.hpp`, `memory_inline_pages.hpp`,
and `page.hpp`.

### Page storage and lookup

The page table uses `BTreeMap<page number, slot>` and stable `Vec` slots
containing `(page number, Page)`. Attributes include read/write/execute,
copy-on-write, `dont_fork`, and a user byte. Storage is arena-backed, owned,
or a shared zero/host page. Owned bytes use `Arc<[u8; 4096]>`; writes use
`Arc::make_mut` to isolate forks. Range operations use `BTreeMap::range`.

A 512-entry direct-mapped page cache stores slots as relaxed `AtomicUsize`
values, allowing lookup through `&self` while preserving `Memory: Sync`.
Hits validate page number and permissions, requiring no invalidation after
copy-on-write, protection changes, or removal. Misses have logarithmic B-tree
cost rather than a guest-controlled hash-collision risk. Arena-free machines
use this path for every load and store; see fork benchmarks.

Only arena pages with non-default attributes need entries. A bitmap tracks
C++-equivalent active pages for `pages_active`, `getrusage`'s `ru_maxrss`, and
bulk-versus-pagewise discard selection.

### Access semantics

Guest arena-window accesses bypass page attributes, matching C++. Other
loads and stores use paging. Host copy, fill, compare, string, and gather
helpers enforce page permissions. `memview` and `memview_mut` use an arena
window or one contiguous gathered buffer; scattered ranges return
`OUT_OF_MEMORY`.

Missing external pages read from the shared zero page and allocate on write.
Owned pages are limited to half `memory_max` with an arena, or the full limit
without one. Attribute-only pages are limited to 64 times the memory limit
in pages; writable zero pages use copy-on-write. `set_page_attr` preserves
C++ update and fault order while scanning existing runs through the page index.

ELF segments load through paging and may lie above the arena. Protections
follow program headers when `protect_segments` is enabled, otherwise RWX.

### Runtime code

`CPU::next_execute_segment` checks target-page execute permission and decodes
adjacent executable pages, limited to `memory_max` bytes. FENCE.I and
`riscv_flush_icache` mark segments stale for rebuilding. The host exit page
is execute-only and decoded on demand.

JIT rebuilds with identical bytes, addresses, and layouts reuse native code
through `jit::reuse`. Changed rebuilds interpret until they remain valid for
`jit_background_slice` executed instructions, then compile through
`DecodedSegment::defer_jit`. New mappings compile normally. This avoids
recompiling repeatedly modified code; one fuzz case previously compiled
754 times in 1,500 instructions, consuming 6 s.

Native checked accesses outside arena windows return `EXIT_SLOW` at the
instruction. The interpreter performs the paged access and finishes the
already-counted block. N-bit accesses bypass paging. Spectre guards and their
configuration are described under JIT.

### Dynamic executables

`rvlinux` loads its fixed
`/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1` with argv
`[interpreter, program, args...]`. The guest opens the program and allowed
libraries under `/lib/riscv64-linux-gnu/`, mapped to
`/usr/riscv64-linux-gnu/lib/`.

`PT_INTERP` cannot choose the host file. Recognized paths ending in
`ld-linux-riscv64-lp64d.so.1` are retained as argv[0] for instruction-count
compatibility; other paths produce a warning and use the configured linker
path. `read_host_file` requires a regular file no larger than 64 MiB.
`tests/interpreter.rs` covers devices, FIFOs, self-references, and canonical
paths. This differs from C++'s guest-selected host path, which could block or
exhaust memory before sandbox setup.

`ld.so` mappings become execute segments when entered. Counts match C++,
including `dlopen`. `FileDescriptors::pinned_mappings` gives each library a
16 MiB slot starting at 1 GiB in list order (`libc.so.6`: `0x43000000`).
The first mapping with no requested address and offset zero uses its slot
when it fits and is free; otherwise ordinary `mmap` placement applies.
Stable addresses permit cache and AOT reuse across programs. A dynamic C++
program measured 27 s with upfront compilation and 0.07 s from cache.
Tests include `pinned_file_mappings` and `dynamic_pinned_libraries`.

Custom page fault/read/write and execute-fault handlers, memory traps, and
`insert_non_owned_memory` remain unimplemented.

## Testing

| Layer | Where | What |
|---|---|---|
| Unit | `#[cfg(test)]` in each module | Decoding, helpers, edge cases |
| Integration | `crates/libriscv/tests/` | Ports of `../libriscv/tests/unit/*.cpp` (Catch2 → `#[test]`); guests built with `testutil::build_and_load`. `linux.rs`: `security.cpp`, Linux cases of `protections.cpp`, threads, signals, the file sandbox, and every system call with random arguments. `paging.rs`: the memory cases of `protections.cpp`, `va_execute.cpp` (code on mprotected and high pages, FENCE.I/`riscv_flush_icache`, constant pools) and `dynamic.cpp` (PIE). `vmcall.rs`: `vmcall.cpp` (the fork case included) and preemption (restoring, clobbering, errors, shared budget, nested under background JIT). `custom.rs`: `custom.cpp`'s custom instruction, and userdata (`with_userdata` included). `filesystem.rs`: the file system sandbox (no filter means no access, normalised paths, special files) without a guest toolchain. `dyncall.rs`: `dyncall.cpp`, adapted to dyncalls being system calls (interpreter and JIT, exactly equal). `native.rs`: `native.cpp` (native heap system calls and the guest C++ containers). `native_rust.rs`: `native_rust.cpp` (the guest Rust containers against a Rust guest, built with `testutil::build_rust`; needs the `riscv64gc-unknown-linux-gnu` target of the pinned toolchain). `rust_collections.rs`: guest-built cycles and deep nesting of the Rust containers, without a Rust guest. `brutal.rs`: `brutal.cpp` (forks, stepping a few instructions at a time, copy-on-write and page cache cases) plus fork snapshots, forks of forks, forking from threads and fork page accounting. `rvv.rs`: `rvv.cpp` (RVV 1.0 in inline assembly, `-march=rv64gcv`; also under the JIT). `rva23.rs`: `rva23.cpp` (Zfhmin, Zimop, Zcmop and Zawrs as raw `.insn`) |
| Example | `examples/asteroids/tests/headless.rs` | The example game played without a terminal: frames at a fixed rate with scripted input, the JIT against the interpreter frame by frame (screens and instruction counts), heap balance across frames. Builds the Rust guest with cargo; skipped without the guest toolchain |
| Differential | `crates/rvlinux/tests/difftest.rs`, `scripts/difftest.sh` | Same ELF under Rust and C++ `rvlinux`: stdout, exit code, instruction count and the `>>> Machine exception` line must match, also at `--fuel` timeouts. Static and dynamic glibc C and C++ programs (stdio, malloc, threads incl. `test_threads.cpp`, condvars, abort, a fault with a signal handler, timeouts; dynamic ones run under `ld.so`), code generated at runtime (mprotect, FENCE.I, pages far outside the arena, `dlopen`, an execute fault), `getrusage`'s page accounting, the prebuilt Rust/Zig/newlib fixtures (Go: output only, its count varies under the oracle too) and glibc CoreMark, and `--libc-fastpath` (the guest of `libc_fastpath.cpp` calling every patched function, the glibc C and C++ programs, timeouts, and faults inside patched functions). Freestanding RV64IM, RV64IMC and RV64IMC+Zcb guests, including one assembly program per dispatcher and compressed-decoder edge case; RV64IMA and RV64IMAC+Zcb for atomics; RV64IM and RV64IMC+Zcb with Zba/Zbb/Zbc/Zbs/Zicond/Zbkb for bit manipulation; RV64IMFD and RV64IMAFDC+Zcb with Zfhmin/Zfa for floating point; CoreMark (soft-float, F/D, F/D+B), whose clock is the instruction counter so its whole output is compared. `rvlinux --call` (a VM call after the program exits). C++'s RVA23 suite (`../libriscv/tests/rva23/`, port of its `run.sh` and `isa/run.sh`): the ten programs built with the RVA23 compiler (`$RVCC`, default `~/riscv/bin/riscv64-unknown-linux-gnu-gcc`) must print what their host builds print, the instruction-level probe must pass, and both must match the reference, also at timeouts inside vector code; skipped without `$RVCC` (`LIBRISCV_REQUIRE_RVA23=1` fails instead), which CI does not have. Each scenario is one test per Rust configuration (`<name>::interp`, `::jit`, `::jit_upfront`), and long fuel loops are split, so the harness runs them concurrently |
| Instruction | ports of `../libriscv/tests/instructions/` | Per-instruction checks incl. RVC (`test_rv32c.cpp` → `tests/rvc.rs`, widened to RV64) |
| ISA | `../libriscv/tests/rva23/` | Ported into `difftest.rs` (`rva23_*`, see above); needs a full RVA23 toolchain |
| Fuzz | `fuzz/` (`cargo fuzz`, a separate workspace; see `fuzz/README.md`) | Port of `../libriscv/fuzz/fuzz.cpp`: `vmfuzzer` (instruction set), `elffuzzer` (ELF loader + Linux system calls, seeded with the prebuilt RV64 ELFs), `nativefuzzer` (native heap and memory system calls; C++ `FUZZ_NAT` without native threads), `syshelperfuzzer` (system call memory helpers). Beyond C++: `linuxfuzzer` (Linux system calls, threads, signals, sandboxed file system), `libcfuzzer` (libc fast path), `forkfuzzer` (forks must continue exactly like their parent), `guestfuzzer` (guest C++ containers), `vmcallfuzzer` (VM calls, nested preemption, dyncalls, system call arguments) and `jitdifffuzzer` (JIT against the interpreter, exact). Built with debug assertions and overflow checks; any panic is a bug. `overnight.sh` runs them all for hours, collecting findings |

Prebuilt guests (Rust, Go, Zig, newlib, tinycc) live in
`../libriscv/tests/unit/elf`; use `testutil::cpp_fixture(name)`.

CI also tests the 4-byte decoder layout (`--no-default-features`, with and
without `jit`). The distribution's glibc is RV64GC, so tests running glibc
guests are ignored (or, for `linux.rs` and `vmcall.rs`, compiled out)
without the A, C and F features; freestanding guests are built for what
the features allow.

CI runs the `test`, `stream` and `lint` jobs on both x86-64
(`ubuntu-24.04`) and AArch64 (`ubuntu-24.04-arm`) hosts, with the C++
reference built natively on each.

CI runs the dynamic STREAM executable (`binaries/STREAM/stream.dyn`)
under Ubuntu's riscv64 glibc: on Linux with the interpreter and `-j`
(the second `-j` run must use the cached system code), and on a FreeBSD
VM (`freebsd` job) with the interpreter and `-j`. Loading cached JIT code
is Linux-only, so on FreeBSD `-j` compiles everything on every run.

Run the C++ reference with `--no-translate --accurate` for interpreter
execution and exact instruction counts.
`scripts/build-cpp-reference.sh` builds the threaded, tailcall and switch
variants into `target/cpp-ref/<mode>/rvlinux` for benchmarking
(`testutil::reference_emulator_variant`).

## Phases

### Phase 0 — Infrastructure (complete)

- [x] Cargo workspace, `forbid(unsafe_code)`, clippy/rustfmt config, CI
- [x] `Width`/`Address`, `MachineError`/`ExceptionKind`, `MachineOptions`, `Instr` formats
- [x] `testutil`: cross-compile guests (cached), run C++ reference, parse output
- [x] `rvlinux` CLI skeleton with output format contract
- [x] Difftest scaffolding (ignored until Phase 2)
- [x] Dispatch decision: pinned nightly + `become`; C++ dispatch variants built and baselined (since 2026-11-16 the opt-in `nightly-dispatch`; stable by default)

### Phase 1 — Minimal machine (RV64I + M, freestanding guests; complete)

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

- VMV.X.S, VCPOP.M, and VFIRST.M with rd = x0 decode as NOPs. C++ writes
  x0, allowing later reads to return nonzero.
- FMIN/FMAX with one signaling NaN depends on the C++ compiler: Clang's
  inlined `minnum` returns the numeric operand, matching Rust, while glibc
  can return quiet NaN. Differential tests exclude this case.
- Fused FP NaN payloads and signs depend on glibc's host FMA versus Rust's
  compiler-builtins implementation. Tests compare NaN classification.
  Results matched on Ryzen 7950X but differed on a GitHub runner.
- Freestanding guests require `-mno-relax` because no crt0 initializes `gp`.
- Arena allocation failure aborts the process; C++ throws `OUT_OF_MEMORY`.
  Memory sizes exceeding the host address space use paging without an arena,
  matching failed C++ arena mapping.

Paging and resource-limit differences:

- Page zero is always a guard page, including without a binary. Empty
  machines retain their arena fast path; C++ uses paging in that case.
- Segments decoded from pages are limited to `memory_max` bytes; larger
  ranges return `OUT_OF_MEMORY`. C++ can decode attribute-only ranges up
  to 64 times the memory limit and exhaust host memory.
- Execute-segment host storage has a separate
  `execute_segment_memory_max` limit, defaulting to
  `max(memory_max, EXECUTE_SEGMENT_MEMORY_MIN)` with a 256 MiB minimum.
  Code plus decoder entries costs about 9 host bytes per code byte,
  supporting at least 28 MiB of code by default. Shared segments count in
  full per machine. Native code is excluded, and background compilation
  temporarily retains another segment copy. `execute_segments_memory`
  reports usage; exceeding the limit returns `OUT_OF_MEMORY`.
- New segments evict covered non-main segments. C++ retains overlapping
  copies, potentially using about 5 GB for a 48 MiB area grown pagewise.
  Evicted slots remain occupied to preserve `MAX_EXECUTE_SEGS` timing.
  Embedders should budget the segment limit in addition to guest memory
  and native code; reduce it for pools of small machines.
- Runtime decoding charges `max(len / 4, 1)` instructions, as C++
  `penalize_execute_segment_decode` (upstream `f601956a`). Newly mapped
  segments and stale rebuilds are charged; load-time and explicit
  `init_execute_area` segments are not. This bounds repeated copying,
  hashing, and decoding of large areas after FENCE.I; a measured
  200-instruction case otherwise consumed 436 ms. Charges reaching the
  limit halt before the next block and do not make deferred compilations
  hot. `MachineOptions::free_decode_bytes` (not in C++, default 0) makes
  that many decoded bytes free per machine. `Machine::penalize` saturates
  instead of wrapping.
- Slow accesses crossing a page boundary use both pages. C++ can access
  beyond an owned heap page; arena crossings use adjacent arena bytes in
  both implementations.
- Discarding the host code page creates a private zeroed copy. C++ can
  zero the shared page for every machine in the process.

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
| `vmcall<MAXI>` / `vmcall<MAXI, false>` | `vmcall_limited` / `try_vmcall` (0 or `u64::MAX`: unlimited, as in C++; the docs warn that a limit of 0 runs untrusted code unbounded) |
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
  the signal return trampoline) is not installed.

### Fork

`Machine::fork_with(&self, options)` ports the forking constructor.
`fork()` reuses the source's options; C++ instead defaults to `MachineOptions{}`.
A fork copies registers, instruction counter and limit, threads, program layout,
mmap free list, and LR/SC reservation. Execute segments and the system call
table are shared; installing a handler makes the table private. Printer,
stdin, and rdtime callbacks reset to defaults. File descriptors, signal
handlers, userdata, and native heap state are not inherited.

`Memory::fork` shares pages through `Arc`, excluding `dont_fork` pages.
Writes on either side use copy-on-write. `minimal_fork` shares no pages;
the default fault handler then supplies zero pages. Custom fault handlers
used by C++ minimal forks are not implemented.

Forks have no arena. Create fork sources with `use_memory_arena: false`.
Forking an arena-backed source with `use_memory_arena` enabled returns
`ILLEGAL_OPERATION`. With it disabled, arena pages represented in C++'s page
table are copied; pages touched only through guest arena accesses are excluded,
matching C++.

Forks are independent snapshots: the source may run, change, or be dropped,
and forks may be forked again. Making shared pages writable or discarding
read-only shared pages creates private copies. C++ requires the source to
outlive its forks and can modify source pages in these cases.

Forks share pending background compilations and receive their results.
Later segments, including the VM-call exit page, can use the process-wide cache.
Native code currently requires an arena, so forks and arena-free sources use
the interpreter. Native page-table access is required before JIT execution can
support them; master-specific N-bit layouts would also be rejected.

### Shared execute segments

Port of C++ `SharedExecuteSegments` (`decoder_cache.cpp`), enabled by default
through `use_shared_execute_segments`. Machines decoding identical code at the
same address share an `Arc<DecodedSegment>` and native code. Keys include
address, length, CRC32-C, execute-only and libc patch settings, backend settings,
and native memory layouts. CRC32-C uses slicing-by-8; C++ uses SSE4.2.

The cache entry remains locked during decoding. Concurrent machines decode
and compile a segment once; upfront compilation waits for an existing
background compilation.

Lookups compare exact code bytes in addition to CRC32-C, preventing collision
substitution. Stale segments are replaced. The cache holds weak references;
segments are freed after their last user, and unused entries are pruned when
the map doubles in size.

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

### Shared read-only data

`memory/shared.rs` ports `shared_rodata.cpp` and
`Memory::prepare_shared_rodata`/`publish_shared_rodata`. Enabled by default
through `use_shared_rodata` with `jit` on Linux x86_64, aarch64, and riscv64.
`rvlinux` disables it because its single machine cannot benefit from sharing.

Machines with matching program bytes and arena sizes map one sealed memfd
(`libriscv-rodata`) over `[0, end)`. `end` is the host-page-aligned read-only
boundary, at most the first writable segment. The first machine publishes
after loading and relocations. Later machines compare every covered byte,
attach before loading, and skip covered copies while retaining page-table
entries through `touch_writable`.

Images use weak references and are released after their last user. The cache
lock covers lookup and insertion only; comparisons and creation occur outside
it. Concurrent identical candidates converge on one image. This supports pools
of independently created machines sharing an `Arc<[u8]>` ELF.

`libriscv_jit_rt::GuestArena` owns an anonymous `MAP_NORESERVE` `memmap2`
mapping exposed as `[u8]`; `RodataImage` owns the memfd. Interpreter, JIT,
and AOT access paths retain the same arena representation. Linux-specific
sealing and fixed-address calls use explicit declarations for supported
architectures. Two inaccessible 1 MiB guards surround the arena as additional
protection; native code still enforces its own access checks.

For discards of at least 256 KiB, Linux mappings reclaim whole anonymous host
pages through `GuestArena::zero_pages`. Attached image pages are excluded,
because discarding them would restore image contents. Without `jit`, the
`Vec<u8>` arena scans nonzero 4 KiB chunks for zeroing. Repeated Go startup
discards measured about 370 ms and 184,000 page faults with vector storage,
versus 12 ms with mapped storage or C++.

Differences from C++:

- Images use `PROT_READ | PROT_WRITE` with `MAP_PRIVATE`. Stores create
  private pages, preserving isolation with virtual paging, `mprotect`,
  and N-bit masking. C++ uses read-only `MAP_SHARED` and disables sharing
  when those modes allow writes. Write/grow/shrink/seal seals prevent
  mutation and truncation-induced `SIGBUS`.
- Image sizes use whole host pages, including hosts with pages larger
  than guest pages.
- Discarding read-only data zeroes private pages, matching paging semantics
  without changing other machines.
- Failed attachment over a loaded arena restores a private mapping from an
  image view acquired before modification. Images use `MAP_NORESERVE`.

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
- Host paths are only looked up through the filters: `openat` and
  `faccessat` need `filter_open`, `fstatat` with a path needs
  `filter_stat` (or `filter_open` without rewriting), `readlinkat` needs
  `filter_readlink`; without them only proxy mode allows it. C++ goes to
  the host when no filter is set (for `openat` too: `permit_filesystem`
  without a filter opened any host path).
- The filters get an absolute, lexically normalised guest path
  (`FileDescriptors::normalize_path`: `.`, `..` and repeated slashes
  collapsed, a trailing slash kept), and the host opens that path or the
  filter's rewrite of it, so `..` cannot escape a prefix check. A path
  relative to a directory descriptor is joined to the directory's guest
  path (`OpenFile::guest_path`) before filtering; C++ filters the
  relative tail and opens it relative to the host directory. Relative
  paths through `AT_FDCWD` are `EBADF` without consulting the filter
  (C++ runs the filter first, which could rewrite them to absolute
  paths), and an empty path is `ENOENT`.
- `openat` refuses (`EACCES`) files that are neither regular files nor
  directories unless `permit_special_files` is set: a FIFO, a terminal or
  `/proc/self/fd/0` could block the host thread in `open` or `read`
  outside the instruction limit. Without `libc` there is no portable
  `O_NONBLOCK`, so the type of what the path resolves to (symbolic links
  followed) is checked before opening, and the opened file's again after.
  `faccessat`'s `R_OK`/`W_OK` probes do not open such files either. C++
  opens anything the filter admits; `rvlinux` sets `permit_special_files`
  (its filter is an exact allow-list that includes `/dev/urandom`).
- `dup` returns a virtual descriptor (C++: a host number the guest cannot
  use), `dup3` cannot replace descriptors (`EPERM`), `pipe2` and
  `getdents64` are `ENOSYS` (a pipe nobody reads would block the host),
  reading fds 1/2 is `EBADF`, and the guest can have at most 1024 files
  open (`EMFILE`).
- `ppoll` answers from the descriptor table (stdin readable, stdout/stderr
  writable, files both, unknown `POLLNVAL`, negative ignored) instead of
  polling host fds, and never waits. As in C++ (`5135210c`), invalid
  timeouts are `EINVAL` and revents is always written back. C++ charges
  the host time a sandboxed `ppoll` with a nonzero timeout took
  (`cafba3e8`), a microsecond per instruction; here nothing waits and
  nothing is charged. `tests/ppoll.rs`.
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

### C++ compatibility behavior

The port preserves these C++ behaviors unless an explicit difference is listed:

- C++'s tail-call and threaded dispatchers disagree observably (see
  "Dispatch"); the port follows the threaded one.
- `R_RISCV_RELATIVE` relocations write the symbol value, not base + addend.
- The precise JALR does not clear bit 0 of the target (an odd target faults);
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
- The compressed slow path (`RV32C_BC_FUNCTION`) does not check for a stale
  segment afterwards, unlike the 4-byte one.
- A misaligned SC fails (`rd = 1`) without faulting and, unlike an
  aligned SC, keeps the reservation; a misaligned LR raises
  `DEADLOCK_REACHED` with data 0.
- Not reproduced: C++ registers an execute segment before decoding it and
  leaves it registered if decoding throws; here it is registered only when
  complete.
- `rvlinux` keeps the NUL terminator of `PT_INTERP` in a dynamic
  executable's `argv[0]` (C++ copies the whole `p_filesz` into a
  `std::string`), so that string takes one byte more on the stack.
- Guest stores inside the arena's write window, and loads in its read
  window, ignore page attributes (`mprotect(PROT_NONE)` does not stop them);
  system calls and other host accesses respect them.
- `memdiscard` (`MADV_DONTNEED`, anonymous `mmap`) zeroes read-only memory
  too, and a run of whole untouched arena pages is wiped without looking at
  the attributes of the pages after the first.
- `clone` reads `ptid`/`tls`/`ctid` from `a4`-`a6`, where glibc's wrapper
  leaves them (the kernel ABI has them in `a2`-`a4`); `set_tid_address`
  truncates the address to an `int`.
- A futex wait that would deadlock (no other runnable thread) writes 8
  zero bytes at the futex address and returns 0. When the running thread
  exits and nothing is suspended, the main thread is resumed even if it is
  blocked (it leaves the blocked list here; see below). Exiting threads
  clear their TID word but do not wake futex waiters; futex timeouts are
  ignored.
- `getrandom` of more than 256 bytes returns -1 (not an errno) and charges
  20k instructions per byte; `uname` says "RISC-V C++ Emulator";
  `readlinkat` with a buffer over 1024 bytes is `ENOMEM`; `nanosleep`
  writes the clamped request back; `sigaction` reports an unset handler as
  `~0xF`.
- EBREAK is not a block end, so a handler's `stop()` only takes effect at
  the next check (C++ `basic.cpp` accepts both 51 and 25 EBREAKs).
- The host-side `membuffer`, `memstring_view` and `try_memmove` take the
  arena fast path after checking only the arena's size, not its read or
  write windows or page attributes (`memstring_view` even below
  `RWREAD_BEGIN`), and `memcpy_from` uses the arena's write window for its
  source. `memcmp` does not check that the start is readable on its arena
  fast path, only the end.
- Not reproduced (C++ bug): the slow path of `memcmp(const void*,
  address_t, size_t)` compares the host byte as a signed `char`, flipping
  the result for bytes above 0x7F; here every path compares unsigned bytes
  (and returns an `Ordering`, since only `memcmp`'s sign is meaningful).
  Across pages it compares a page's stretch at a time where C++ looks the
  page up for every byte (same result and faults).
  The guest-to-guest `memcmp` behind the native memcmp system call and
  the libc fast path returns the difference of the first differing
  (unsigned) bytes, what C++'s slow path returns; its fast paths return
  the host `std::memcmp`'s value, whose magnitude depends on the host libc.
  Outside the arena it compares a page-bounded stretch at a time, not a
  byte at a time with two page lookups per byte as C++ does (same result
  and faults): one native memcmp of up to 512 MiB runs before its penalty
  is charged, so the instruction limit cannot bound it.
- Not reproduced (C++ bugs): killing a suspended thread (`tgkill`) leaves a dangling pointer in C++'s
  scheduler lists, here the thread is removed from them; a signal number
  above 64 throws `std::out_of_range`, here it is `ILLEGAL_OPERATION`.
- Not reproduced (C++ bug, as far as the port's reading of `threads.hpp`
  goes): the main thread resumed while blocked (above) stays in C++'s
  blocked list, so it gains an entry each time it blocks again (a guest
  grows the list, and the host work of every exit and wakeup that scans
  it, without bound) and a futex wake can queue it while it runs, giving
  it two schedulable instances. Here resuming a thread takes it off both
  lists, a thread is in at most one of them, and queueing a listed thread
  is an `INVALID_PROGRAM` error. Programs that never reach that state
  schedule exactly as in C++.
- Not reproduced (C++ leak, as far as the port can tell without the C++
  tree): per-thread signal state (altstack, the registers a handler
  interrupted) stays in C++'s map after its thread exits, and TIDs are not
  reused, so a guest grows host memory (about 1 KB a thread, outside
  `memory_max`) by creating threads that touch it. Here a thread's state
  goes when it exits, and `sigaltstack` without a new stack (and
  `rt_sigreturn` without a delivered signal) reads the default state
  without creating any. The guest sees the same values.
- Not reproduced (C++ bugs): `nanosleep` and `clock_nanosleep` charge one
  instruction per microsecond requested, computed in C++ in an `int64_t`
  that overflows (undefined) for requests of 292 years and more; here the
  charge saturates (`u64::MAX / 1000` at most), so `tv_sec = 2^55` is not
  free while the host sleeps. Smaller requests charge exactly what C++
  does. C++ clamps only the seconds to `NANOSLEEP_MAX_SECONDS`, so a host
  sleep could last almost two; here the whole sleep is clamped, while the
  request `nanosleep` writes back is still C++'s (seconds clamped).

### Native heap and libc fast path

`native/` ports `native_heap.hpp` and `native_libc.cpp` (not
`native_threads.cpp`):

- `native/heap.rs`: the allocator (`Arena`), step for step, so the same
  calls give the same addresses (`tests/heap.rs` replays 200k random
  calls, with and without a chunk cap, against the results C++ gave). Its errors are returned with
  C++'s kind, message and data; the search statistics C++ only keeps with
  `RISCV_ARENA_DEBUG` (which its unit tests enable) are always kept. The
  inline metadata storage (`ArenaInlineVector`) is kept, since
  `metadata_on_heap` reports it.
- `native/mod.rs`: `Machine::setup_native_heap` (malloc, calloc,
  realloc, free, meminfo), `setup_native_memory` (memcpy, memset, memmove,
  memcmp, strlen, strncmp, the reserved call, backtrace), `arena()`,
  `transfer_arena_from`, with C++'s penalties. Forks do not get the arena
  (a C++ TODO). Guest-to-guest copies use `Memory::memcpy_within`
  (C++ `memcpy(dst, machine, src, len)` with the same machine), which
  copies overlapping arena ranges like `memmove` (C++'s `std::memcpy`
  leaves that undefined). Outside the arena it copies a page-bounded
  chunk at a time where C++ loads and stores a byte or word at a time,
  with the same memory, faults and created pages (overlap and C++'s
  arena shortcut included; tested against the loop): one native memcpy of
  up to 512 MiB runs before its penalty is charged. The native memmove's
  backward copy (`Memory::memmove_backward`, when `dst >= src`) does the
  same: page-bounded chunks from the end, matching C++'s loop of word or
  byte loads and stores from the end (tested against it). Where the
  loop's stores reach its later loads (`dst` just ahead of `src` going
  forwards, or just behind it modulo 2^64 going backwards), the bytes
  repeat with that period; a chunk repeats them the same way rather than
  shrinking to the distance (a byte at a time for `memmove(-1, 0, n)`).
  A backward chunk's end may also lie wholly in the arena's read (source)
  or write (destination) window, where the loop's loads and stores never
  fault and skip the page table, so copies between the arena and pages
  outside it go a chunk at a time too.
- `native/backtrace.rs`: `Memory::lookup`/`print_backtrace` as
  `Machine::lookup`/`print_backtrace`, without demangling (C++ demangles
  with `__cxa_demangle` when built with `DEMANGLE_ENABLED`).
- `native/libc.rs`: the libc fast path (`MachineOptions::libc_fastpath`,
  `rvlinux --libc-fastpath`). Known libc functions in the program's
  initial execute segments become block-ending EBREAK entries carrying a
  function id; the EBREAK handler runs them natively and returns to `ra`,
  and chains to the previous handler for other EBREAKs. Breakpoints
  (`MachineOptions::ebreak_locations`, C++ `install_ebreak_for`) use the
  same block-ending patch (`DecodedSegment::create_block_ending_entry_at`).

Differences from C++:

- The EBREAK handler chain is per machine (C++: statics, so once any
  machine installs the fast path, every machine of the process has its
  handler). Forks keep their master's chain.
- Shared execute segments are keyed on whether the libc fast path patched
  them, and segments with breakpoints are not shared. (C++ keys on neither,
  so a machine can get another machine's patched segment.)
- The JIT never compiles SYSTEM entries (`DecoderEntry::is_system`), so it
  leaves native code in front of a patched entry and the interpreter runs
  it. (C++ asmjit translates the segment before the patches are
  applied.)
- Not reproduced (C++ bugs, found by `libcfuzzer`; each let a guest loop
  past the instruction limit forever):
  - With the C extension, C++ gives the entries in front of a patched
    function (or a breakpoint) in its block an instruction count of 0
    (`create_block_ending_entry_at`'s TODO), so code falling through into
    it was not counted. Here they count the instructions up to and
    including the patched entry. The patched entry itself still counts 0,
    as in C++, so calls count exactly as there.
  - C++ only finds the block start through entries in front of the patch,
    stopping at the second half of a preceding 4-byte instruction (the
    function was then never recognised). Here the search steps over it.
  - A patched function returning to its own entry (`ra` = its address,
    which no real caller does) falls through to the next instruction,
    since PC did not change, and the call counts nothing. Here such a call
    counts one instruction, so a cycle of patched functions cannot run
    for free.

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

### Guest containers

`guest/` provides host-side RV64 layouts for C++ standard containers and Rust
collections. See the `guest` module documentation for API details.

C++ mirrors implement `GuestPod`: `GuestStdString`, `GuestStdVector<T>`,
`GuestStdUnorderedMap<K, V, H>`, and `GuestStdVariant<E>`. Copies are shallow.
Mutations use `GuestPtr<T>` at the guest address, preserving string SSO and
map self-pointers. Methods take the machine without retaining a borrow.

`GuestType` defines freeing and relocation, `GuestInit` constructs from host
values, `GuestHashKey`/`GuestKeyArg` implement guest hashing and equality,
and `ToHost` copies into host values. Hash policy types `StdHash` and
`CustomHash` replace C++'s `CacheHashCode` template default. `guest_variant!`
declares alternatives; `VariantAlternative` provides typed access. `emplace`
requires the caller to select an alternative explicitly.
`ScopedArenaObject<T>` owns a native-heap value and requires explicit
`free(machine)` because `Drop` cannot access the machine.

Rust scripting types include `GuestRustString`, `GuestRustStr`, `GuestRustVec`,
`GuestRustSlice`, sized and unsized boxes, `GuestRustEnum`, and
`GuestRustAttributes`/`GuestRustAttr`. Layouts match rustc's
`riscv64gc-unknown-linux-gnu` output, including `{ cap, ptr, len }` vectors,
verified by `tests/native_rust.rs`.

- Value methods update host copies. `GuestPtr` methods read, modify, and
  write guest objects in place; `GuestPtr::modify` provides the general form.
  Names follow `std`, including `push`, `pop`, `reserve`, `get`, and map methods.
  Guest memory operations take `Machine` and return `Result`.
- `GuestRustBox<T>` owns a guest value and dereferences to its `GuestPtr`.
  VM calls pass its address as a `Box<T>`, `&T`, or pointer argument.
- Collections require explicit `free`. Ownership-returning methods such as
  `pop`, `remove`, and `into_raw` transfer freeing responsibility to callers.
- Recursive freeing and host copying stop at `MAX_NESTING` (128), returning
  an error for cycles or excessive depth instead of overflowing the host stack.
- Host copies share a top-level `MAX_HOST_COPY` budget, default 64 MiB
  (4 × `MAX_BYTES`), overridable by `with_host_copy_limit`. Text, elements,
  boxes, and entries charge their allocations, with at least one byte per
  item. Exhaustion returns `IllegalOperation`. Per-container limits alone
  cannot bound aliased or exponentially shared guest structures. Hosts
  retaining copies across calls must enforce aggregate limits; direct copies
  of zero-copy arguments also require host limits. Asteroids uses
  `MAX_LOG_BYTES` for this purpose. Tests cover both nesting and copy budgets
  in `tests/rust_collections.rs`.
- Allocation follows `RawVec`: construction uses exact capacity, growth
  doubles from minimum capacities, and zero-sized elements never allocate
  and store capacity zero.
- `guest_rust_enum!` declares guest alternatives and optionally a host enum
  through `=> HostName`. Unit variants use `()`. Tags must be at least as
  wide as payload alignment, and alternatives must not contain self-pointers;
  `PAYLOAD_OFFSET` enforces these constraints at compile time.

Rust guest host calls use counted dyncalls, with no `a7` and only declared
register operands. Native heap and memory operations are ordinary system call
handlers; `Machine::syscall_handler` supports chaining for instrumentation.
A guest `&str` can be decoded as validated borrowed text or an unread
`GuestRustStr`. `Machine::with_userdata` temporarily moves state out to allow
simultaneous state updates and guest-memory borrows without allocation.

`examples/asteroids` demonstrates this API with a Rust engine and RV64GCB game
script. Shared `src/api.rs` numbers define the ABI. `host_functions!` derives
register counts and safe wrappers over naked dyncall stubs;
`native_heap!`/`native_memory!` forward allocation and memory operations.
Headless tests compare every frame and instruction count between interpreter
and JIT. `rustguestfuzzer` exercises collection operations on supplied bytes.

### Phase 3 — Embedding API

- [x] `install_syscall_handler`, typed `sysarg`/`sysargs`/`return_value`, `set_result(_or_error)` (`sysargs.rs`: `SysArg`/`SysResult` traits, tuples of up to 8; `ByPointer<T>` for C++'s pointer-to-struct arguments; Rust guests' `&str`, not in C++)
- [x] `address_of` (ELF symbols), `vmcall`, `preempt` (restoring and clobbering), `setup_call`, `default_exit_function`, `rvlinux --call` (`machine_vmcall.hpp`, `machine_inline.hpp`, `CPU::preempt_internal`); see "VM calls"
- [ ] Call-depth handling
- [ ] `PreparedCall` (`prepared_call.hpp`)
- [x] Guest memory views, `copy_to/from_guest`, `memstring`, `stack_push` (`memory/helpers.rs`: `Buffer`/`membuffer`, `strlen`, `memstring_view`, `memview`/`memview_mut` with `maxlen`, typed `memarray`/`memspan`/`try_memarray` as `GuestSlice`/`GuestSliceMut` over `GuestPod`, `try_memmove`, both `memcmp`s, machine-to-machine `memcpy_from`, `read_value`/`write_value`; `stack_push_value`)
- [x] Shared ELF: `Machine::new(Arc<[u8]>, options)` and `Memory::binary() -> &Arc<[u8]>`, so many machines run one ELF without copying it. C++'s `Machine(std::string_view)` borrows the caller's bytes, which must outlive the machine; here they are shared by reference count instead (forks share the machine's)
- [x] Printer / stdin / rdtime hooks
- [x] Userdata (`set_userdata`/`get_userdata`; `with_userdata`, not in C++), custom instructions (`on_unimplemented_instruction`); see "Architecture decisions"
- [x] Fork with CoW (`machine.cpp`, `cpu.cpp`, `memory.cpp` `machine_loader`), page caches; see "Fork". (C++ `Memory::reset()` is empty: nothing to port)
- [ ] Serialize/deserialize (`serialize.cpp`)
- [ ] Ports of `tests/unit/{dynamic,examples,checksum,pcrel,png}.cpp` (`exceptions.cpp`: done, `tests/exceptions.rs`) (`rvbuffer.cpp` and `custom.cpp`'s "Take custom system arguments": done, `tests/guest_memory.rs`; "Custom instruction": done, `tests/custom.rs`) (`dyncall.cpp`: done, `tests/dyncall.rs`) (`crc32c.cpp`: the CRC32-C case, in `decoder_cache.rs`) (`vmcall.cpp`: done, `tests/vmcall.rs`; `brutal.cpp`: done except memory traps, `tests/brutal.rs`)

### Phase 4 — Performance

- [x] Benchmark harness: CoreMark (`crates/rvlinux/benches/coremark.rs`) vs the C++ reference, `target/cpp-ref/{threaded,tailcall,asmjit}`; VM calls (`crates/libriscv/benches/vmcall.rs`)
- [ ] Benchmark harness: dhrystone, STREAM
- [x] Specialised bytecodes (`decode_bytecodes.cpp`, `threaded_rewriter.cpp`, `bytecode_impl.cpp`, `threaded_bytecodes.hpp`); see "Bytecodes"
- [x] `extern "rust-preserve-none"` handlers (`nightly-dispatch`)
- [x] Stable-Rust dispatch by default, `nightly-dispatch` opt-in, one set of handlers (`dispatch/abi.rs`; `dispatch_slice`, `rvlinux --dispatch-slice`); see "Dispatch"
- [x] Stable toolchain pin (`rust-toolchain.toml`: 1.98), and CI for both dispatch modes; the pinned nightly only for `nightly-dispatch`, AOT and fuzzing (`fuzz/rust-toolchain.toml`, CI `NIGHTLY`)
- [x] Instruction fusion (`rvi_fuse.cpp`): nothing to port, C++ never calls `try_fuse`
- [x] Fused bytecode pairs (`fused_bytecodes.hpp`, `fused_pairs.inc`, `fuse_bytecodes`, `bytecode_bodies.hpp`); see "Fused pairs"
- [x] Shared execute segments across machines and forks, background JIT results reaching every holder (`decoder_cache.cpp` `SharedExecuteSegments`, `util/crc32.hpp`); see "Shared execute segments"
- [x] Shared read-only data (`shared_rodata.cpp`), copy-on-write, with the `jit` feature; see "Shared read-only data"
- [ ] Bounds-check elimination audit (`cargo asm`)
- [x] Cranelift JIT (`jit` feature) replacing asmjit; see "JIT". RV64IMFDC, Zcb, B (no CLMUL), Zicond; A, CSRs, Zfa, Zfhmin and V stay interpreted
- [x] JIT: background compilation (`jit_background`, `rvlinux -B`/`--no-background`)
- [x] JIT: F/D (FCVT to integers exact, NaNs the host's)
- [x] JIT: code cache, Cranelift output saved as an object to link in or a shared object to load (`jit_cache`, `rvlinux --jit-save-object`/`--jit-save-shared`/`--jit-load`, `RVLINUX_JIT_EMBED`; the role of C++'s translation cache and embedded translations); see "Code cache" under "JIT"
- [x] JIT: automatic N-bit address space (`jit_automatic_nbit_address_space`, `rvlinux --nbit-address-space`; port of `nbit_arena.cpp`)
- [x] JIT: native calls between regions (asmjit `emit_direct_call`, shared with AOT in `codegen::calls`); see "Native calls" under "JIT"
- [x] JIT: dyncall handlers called from native code (asmjit `emit_dyncall`); see "Dyncalls" under "JIT"
- [x] AOT: translation to safe Rust (`aot`/`aot-dylib` features, `rvlinux --aot-save`/`--aot-save-shared`/`--aot-load`, `RVLINUX_AOT_EMBED`), structured control flow, region-to-region `become`, native calls (inlined when self-contained, devirtualized for address-taken functions); see "AOT"
- [x] JIT: AArch64 hosts (instruction-cache maintenance in `libriscv-jit-rt`; code cache with `R_AARCH64_PREL64`)
- [ ] JIT: lazy compilation. Lazy compilation matters most for dynamic executables: `libc.so`'s whole executable mapping (1.5 MB) becomes one segment, and compiling it up front takes ~10 s (release)

### Phase 5 — Remaining features

- [x] V extension (`rvv.hpp`, `rvv_instr.cpp`, `rvv_registers.hpp`): RVV 1.0
  with Zvbb and Zvfhmin, VLEN = 256, the vector CSRs (`handle_vector_csr`);
  tests: `rvv.cpp`, `rva23.cpp` (`tests/rvv.rs`, `tests/rva23.rs`) and the
  RVA23 suite (`difftest.rs`). Every vector instruction is a slow-path
  (FUNCTION) instruction, as in C++; the JIT and AOT leave them to the
  interpreter. The C++ handlers are generic lambdas over SEW; the `int_sew!`,
  `widen_sew!` and `fp_sew!` macros expand the same bodies per SEW, and the
  fixed-point rounding helpers work in `i128` instead of per element type.
  Unit-stride and whole-register transfers copy straight between guest memory
  and the register file, so a fault midway leaves the same partial state as
  C++'s `memcpy`. VMV.X.S, VCPOP.M and VFIRST.M with rd = x0 decode as
  NOPs (C++ writes x0; see "Known differences from the oracle").
  Performance is not a goal yet
- [ ] Native threads (`native_threads.cpp`); POSIX threads are done (Phase 2)
- [x] Native heap and libc fast paths (`native_heap.hpp`, `native_libc.cpp`; see "Native heap and libc fast path"), breakpoints (`ebreak_locations`); tests: `heaptest.cpp`, `heap_invariants.cpp`, `libc_fastpath.cpp` (`tests/heap.rs`, `tests/libc_fastpath.rs`), `native.cpp` (`tests/native.rs`, with the container cases), `exceptions.cpp`, difftests with `--libc-fastpath`
- [x] Guest datatypes, C++ and Rust containers (`guest_datatypes.hpp`, `guest/guest_{common,arena_object,cpp_*,rust_*}.hpp`); see "Guest containers". Tests: `native.cpp` (`tests/native.rs`), `native_rust.cpp` (`tests/native_rust.rs`, Rust guests built by `testutil::build_rust`, the port of `rustbuilder.cpp`)
- [x] Dynamic calls (`dyncall.hpp`): the encoding; dyncalls are system calls numbered by their index, counted exactly in the interpreter and the JIT (see "JIT")
- [ ] Disassembler (`rv_printer.hpp`, `rvv_printer.hpp`, `rv_disassembly.hpp`)
- [ ] CLI debugger (`debug.cpp`, `debug.hpp`)
- [ ] GDB RSP server (`rsp_server.hpp`, `linux/rsp_server.hpp`)
- [x] Virtual paging (`memory_helpers_paging.hpp`, `memory_inline_pages.hpp`, `page.hpp`, paging parts of `memory.cpp`/`memory_rw.cpp`/`cpu.cpp`); see "Virtual paging" (page caches included)
- [ ] Memory traps (`RISCV_MEMORY_TRAPS`)
- [ ] Sockets, epoll, poll, select (`posix/socket_calls.cpp`, `linux/syscalls_{epoll,poll,select}.cpp`)
- [x] Fuzzing harness (`fuzz/`; `nativefuzzer` without native threads), plus targets C++ does not have (Linux personality, libc fast path, forks, guest containers, VM calls, JIT differential)

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
| `registers.hpp`, `rvv_registers.hpp` | `registers.rs`, `rvv_registers.rs` | 1–2, 5 | ✅ integer, FP (raw bits) + FCSR (the `fp64reg` views are in `isa/rvfd.rs`), vector |
| `cpu.hpp`, `cpu.cpp`, `cpu_inline.hpp` | `cpu.rs` | 1 | ✅ (`create_block_ending_entry_at`, `install_ebreak_for`: `decoder_cache.rs`; other debugger helpers: Phase 5) |
| `tailcall_dispatch.cpp` | `dispatch.rs`, `dispatch/abi.rs` | 1, 4 | ✅ |
| `cpu_dispatch.cpp`, `threaded_dispatch.cpp`, `cpu_inaccurate_dispatch.cpp`, `threaded_inaccurate_dispatch.cpp` | — | — | superseded by tail-call dispatch; `cpu_dispatch.cpp` defines the observable semantics (see "Dispatch") |
| `machine.hpp`, `machine.cpp`, `machine_inline.hpp` | `machine.rs`, `sysargs.rs` | 1–3 | Phase 1 part ✅ (`system()`, syscalls, simulate), fork ✅, typed `sysargs`/`set_result` ✅ |
| `machine_vmcall.hpp`, `prepared_call.hpp` | `vmcall.rs` | 3 | ✅ `vmcall`/`preempt` (`PreparedCall`: not yet) |
| `instruction_list.hpp`, `instr_decoding.inc`, `instr_helpers.hpp`, `safe_instr_loader.hpp` | `decode.rs` | 1–2 | ✅ RV64IMAFDCV + Zcb + B + Zfhmin/Zfa (`safe_instr_loader.hpp` is in `decoder_cache.rs`) |
| `rvi_instr.cpp`, `rv64i.cpp` | `isa/rvi.rs` | 1–2 | ✅ I + M + Zba/Zbb/Zbc/Zbs/Zicond/Zbkb (Zbkb's ZIP/UNZIP are RV32-only and unimplemented, as in C++) |
| `rvc.hpp`, `rvc_instr.cpp` | `instruction.rs` (`CInstr`), `isa/rvc.rs` | 2 | ✅ RV64 C + Zcb |
| `rva.hpp`, `rva_instr.cpp` | `isa/rva.rs` | 2 | ✅ RV64 (the reservation is copied on fork; on deserialize: Phase 3) |
| `rvfd.hpp`, `rvfd_util.hpp`, `rvf_instr.cpp`, `fp16.hpp` | `isa/rvfd.rs` | 2 | ✅ F, D, Zfhmin, Zfa (RV64), NaN-boxing under `debug-checks` (FCSR emulation: not ported) |
| `rvv.hpp`, `rvv_instr.cpp` | `isa/rvv.rs` | 5 | ✅ RVV 1.0, Zvbb, Zvfhmin (printers: with the disassembler) |
| `rvi_fuse.cpp` | — | — | dead code in C++ (`try_fuse` is never called) |
| `instruction_counter.hpp` | `dispatch.rs` (`Counter`) | 1 | ✅ |
| `decoder_cache.*`, `decoded_exec_segment.hpp` | `decoder_cache.rs` | 1, 4 | ✅ incl. shared segments, EBREAK locations and the libc fast path patches |
| `decode_bytecodes.cpp`, `threaded_rewriter.cpp` | `decode.rs` (`fast_handler`, `cpp_entry_bits`) | 1–4 | ✅ RV64IMAFDC + Zcb + B |
| `bytecode_impl.cpp`, `threaded_bytecodes.hpp` | `bytecode.rs` (slow-path and system bytecodes: `dispatch.rs`) | 4 | ✅ |
| `fused_bytecodes.hpp`, `fused_pairs.inc`, `bytecode_bodies.hpp` | `fused.rs`, `bytecode.rs` (`Bytecode`, `First`), `decoder_cache.rs` (`fuse_bytecodes`, `unfuse_entry_before`) | 4 | ✅ all 128 pairs |
| `bytecode_dispatch.cpp`, `threaded_bytecode_array.hpp` | — | — | superseded by tail-call dispatch |
| `shared_rodata.*` | `memory/shared.rs`, `libriscv-jit-rt/src/arena.rs` | 4 | ✅ copy-on-write image, also with paging and N-bit arenas |
| `machine_defaults.cpp`, `posix/minimal.cpp` | `machine.rs`, `syscalls/minimal.rs` | 1–2 | ✅ |
| `linux/system_calls.cpp`, `linux/syscalls_*.cpp` | `syscalls/linux/` | 2, 5 | ✅ incl. mman, poll, select (epoll, eventfd: Phase 5) |
| `posix/filedesc.hpp` | `syscalls/fds.rs` | 2 | ✅ (`std::fs::File`s, no raw fds) |
| `posix/signals.*` | `signals.rs` | 2 | ✅ |
| `posix/socket_calls.cpp` | `syscalls/socket.rs` | 5 | |
| `posix/threads.cpp`, `threads.hpp`, `native_threads.cpp` | `threads.rs` | 2, 5 | ✅ POSIX threads (native threads: Phase 5) |
| `native_heap.hpp`, `native_libc.cpp` | `native/` | 5 | ✅ (native threads: not ported) |
| `serialize.cpp` | `serialize.rs` | 3 | |
| `debug.*` | `debug.rs` | 5 | |
| `rv_printer.hpp`, `rvv_printer.hpp`, `rv_disassembly.hpp` | `disasm/` | 5 | |
| `rsp_server.hpp`, `linux/rsp_server.hpp` | `rsp.rs` | 5 | |
| `dyncall.hpp` | `dyncall.rs` | 5 | ✅ (JIT side in `jit/`, see "JIT") |
| `guest/`, `guest_datatypes.hpp` | `guest/` | 5 | ✅ C++ and Rust containers |
| `livepatch.hpp` | — | — | evaluate (mostly JIT-related) |
| `asmjit/aj_translate.cpp`, `aj_emit.cpp`, `aj_api.cpp` | `jit/` (Cranelift), `libriscv-jit-rt` | 4 | ✅ replaced, not ported |
| `tr_translate.cpp` (translation cache, embedded translations) | `jit/cache.rs`, `libriscv-jit-rt/src/linked.rs` | 4 | ✅ replaced: caches the Cranelift JIT's code, translates nothing |
| `tr_emit.cpp`, `tr_translate.cpp`, `tr_compiler.cpp`, `tr_api.cpp` (translation to C) | `aot/`, `codegen/`, `native_code.rs`, `libriscv-jit-rt/src/aot.rs` | 4 | ✅ replaced: translation to safe Rust |
| `util/auxvec.hpp`, `Machine::setup_linux`/`setup_argv` | `linux.rs` | 2 | ✅ |
| `util/crc32*.{hpp,cpp}` | `decoder_cache.rs` (`crc32c`) | 4 | ✅ CRC32-C (plain CRC32: not needed yet) |
| `util/buffer.hpp` | `memory/helpers.rs` | 3 | ✅ `Buffer` |
| `util/function.hpp`, `util/threadpool.h`, `util/load_binary_file.hpp` | `std` | — | not needed |
| `emulator/src/main.cpp` | `crates/rvlinux` | 0–2, 5 | ✅ static and dynamic Linux guests (`full_linux_guest`), `--call` (without the system call 500 symbol helper, which needs userdata); debugger, GDB: later |
| `tr_tcc.cpp`, `tr_emit_rvc.cpp` (RV32 C), `amd64/`, `win32/`, `rv128i*`, `rva128_instr.cpp` | — | — | out of scope |
