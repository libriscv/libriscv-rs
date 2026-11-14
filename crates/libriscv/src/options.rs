//! Machine construction options — port of C++ `MachineOptions<W>` (common.hpp).
//!
//! Only interpreter-relevant options are carried over. Binary-translation
//! and libtcc options are intentionally absent (see PORTING.md, "Scope");
//! the asmjit ones are replaced by `jit` (the Cranelift JIT).
//! Add fields here as the features they control get ported.

/// Options passed when constructing a machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineOptions {
    /// Maximum guest memory, rounded down to the page size.
    pub memory_max: u64,
    /// Virtual memory reserved for the main stack at construction.
    pub stack_size: u32,
    /// Load the binary as a RISC-V ELF at construction.
    pub load_program: bool,
    /// Apply page protections from the ELF program headers.
    pub protect_segments: bool,
    /// Allow read-write-execute segments.
    pub allow_write_exec_segment: bool,
    /// Enforce execute-only segments (X without R).
    pub enforce_exec_only: bool,
    /// Ignore the `.text` section and load all executable segments as-is.
    pub ignore_text_section: bool,
    /// Print verbose loader information.
    pub verbose_loader: bool,
    /// Skip copy-on-write page sharing when forking.
    pub minimal_fork: bool,
    /// Use a linear memory arena for main memory.
    pub use_memory_arena: bool,
    /// Preserve vector registers when copying register state.
    pub preserve_vector_registers: bool,
    /// Compile execute segments to native code with the Cranelift JIT.
    /// Requires the `jit` feature; without it, `Machine::new` fails with
    /// `FEATURE_DISABLED`. On hosts the JIT doesn't support yet, segments
    /// are silently interpreted.
    pub jit: bool,
    /// With `jit`, compile execute segments on background threads instead of
    /// when they are created (C++ `asmjit_background_callback`). The
    /// interpreter runs the segment meanwhile, and switches to native code
    /// once it's ready; results and instruction counts are the same either
    /// way. While a compilation is pending, `simulate` runs in slices of
    /// `jit_background_slice` instructions, checking for finished code in
    /// between. Short-lived programs are better off without the JIT.
    pub jit_background: bool,
    /// Instructions between checks for finished background compilations
    /// (at least 1). Smaller slices switch to native code sooner.
    pub jit_background_slice: u64,
    /// With `jit` (and in translations made with `aot_output`), mask guest
    /// addresses with `arena size - 1` in native code instead of
    /// bounds-checking them (C++
    /// `translate_automatic_nbit_address_space`). Much cheaper loads and
    /// stores, but native code then reads and writes anywhere in the arena,
    /// read-only data and the first page included, and accesses outside it
    /// wrap around instead of going to the page table. The mask needs a
    /// power-of-two arena, so `memory_max` is rounded up to one (the
    /// default 4000 MiB of `rvlinux` becomes 4 GiB); rounding up by more
    /// than 256 MiB fails `Machine::new` with `OutOfMemory`. The
    /// interpreter is otherwise unaffected. Translations are only used by
    /// machines with the same setting.
    pub jit_automatic_nbit_address_space: bool,
    /// With `jit` (and in translations made with `aot_output`), let native
    /// code run without counting instructions or checking the instruction
    /// limit (C++ `translate_ignore_instruction_limit`): loops in native
    /// code then run until they leave it, however long that takes, so the
    /// program needs another way to be stopped. The interpreter still
    /// counts the blocks it runs and stops at the limit, so the counter
    /// becomes an undercount, only good for telling progress (`rvlinux`
    /// sets this unless `--accurate`, like C++). Native code leaves at
    /// system calls, so `stop()` still works. Translations are only used
    /// by machines with the same setting.
    pub jit_ignore_instruction_limit: bool,
    /// Use JIT code compiled by an earlier run for the execute segments it
    /// was compiled for (default on): code linked into the program, and
    /// code from shared objects loaded with
    /// [`jit_cache::load_shared_object`](crate::jit_cache::load_shared_object).
    /// Such code is used instead of compiling, also without `jit`.
    /// Results and instruction counts are the JIT's. Without the `jit`
    /// feature, there is never any.
    pub jit_cache: bool,
    /// With `jit`, record the code of every segment the JIT compiles here,
    /// to save it for later runs (see [`JitCacheOutput`]). The machine
    /// then compiles all its segments itself, instead of using cached code
    /// or another machine's shared segments.
    ///
    /// [`JitCacheOutput`]: crate::jit_cache::JitCacheOutput
    #[cfg(feature = "jit")]
    pub jit_cache_output: Option<crate::jit_cache::JitCacheOutput>,
    /// Run execute segments with code translated to Rust ahead of time
    /// (default on), where a translation for them is registered: compiled
    /// into the program ([`aot::register`]) or loaded from a shared object
    /// ([`aot::load_shared_object`], feature `aot-dylib`). A translation is used instead of the
    /// interpreter and the JIT, for the segments and the memory layout it
    /// was made for. Results and instruction counts are the interpreter's.
    /// Without the `aot` feature, there is never any.
    ///
    /// [`aot::register`]: crate::aot::register
    /// [`aot::load_shared_object`]: crate::aot::load_shared_object
    pub aot: bool,
    /// Translate every execute segment the machine creates to Rust and
    /// record it here, to write it as source or a shared object for later
    /// runs (see [`AotOutput`](crate::aot::AotOutput)). The machine then
    /// creates all its segments itself, instead of using translations or
    /// another machine's shared segments. It runs normally otherwise
    /// (interpreted, or with the JIT).
    #[cfg(feature = "aot")]
    pub aot_output: Option<crate::aot::AotOutput>,
    /// The ELF symbol of a program-provided function to use as the exit
    /// address of VM calls (the return address [`Machine::vmcall`] sets up)
    /// instead of the injected host code page (C++
    /// `default_exit_function`). The function should end the call with the
    /// STOP instruction (`.insn i SYSTEM, 0, x0, x0, 0x7ff`). Returning to
    /// code in the program's own execute segment avoids a segment switch.
    /// If the symbol isn't found, the host page is used. Like C++, the host
    /// page (and with it the signal return trampoline) is then not
    /// installed.
    ///
    /// [`Machine::vmcall`]: crate::Machine::vmcall
    pub default_exit_function: Option<String>,
    /// Share decoded execute segments, and their JIT-compiled code, between
    /// all machines of the process that decode the same code at the same
    /// address (C++ `use_shared_execute_segments`, default on). Forks share
    /// their master's segments either way.
    pub use_shared_execute_segments: bool,
    /// Share the program's text and rodata (the arena below the first
    /// writable segment, in whole pages) in host memory with every other
    /// machine of the process loaded from the same program into an arena
    /// of the same size (C++ `use_shared_rodata`, default on). Needs the
    /// `jit` feature (its runtime crate owns the mapping) and Linux,
    /// otherwise it does nothing; the machine doesn't have to use the JIT.
    /// Unlike C++, the image is mapped copy-on-write, so it also works with
    /// paging and with N-bit address masking: a store that reaches one of
    /// its pages (the guest made it writable, a masked store) gets a
    /// private copy of that page, invisible to other machines.
    pub use_shared_rodata: bool,
    /// Breakpoints: EBREAKs installed in the decoder cache of every
    /// execute segment containing them (guest memory is unchanged), which
    /// run the EBREAK system call handler (C++ `ebreak_locations`).
    /// Segments with breakpoints are not shared with other machines.
    pub ebreak_locations: Vec<crate::EbreakLocation>,
    /// Hot-patch known libc functions (memcpy, memset, memmove, memcmp,
    /// memchr, rawmemchr, strlen, strnlen, strcmp, strncmp, strcpy, stpcpy,
    /// strncpy, strchr, strchrnul and their aliases, found by ELF symbol)
    /// in the program's initial execute segment with native
    /// implementations, charged like the native memory system calls (C++
    /// `libc_fastpath`). Only the decoder cache is patched; other EBREAKs
    /// still reach the EBREAK handler, whenever it is installed.
    pub libc_fastpath: bool,
}

impl Default for MachineOptions {
    fn default() -> Self {
        Self {
            memory_max: 64 << 20,
            stack_size: 1 << 20,
            load_program: true,
            protect_segments: true,
            allow_write_exec_segment: false,
            enforce_exec_only: false,
            ignore_text_section: false,
            verbose_loader: false,
            minimal_fork: false,
            use_memory_arena: true,
            preserve_vector_registers: true,
            jit: false,
            jit_background: false,
            jit_background_slice: crate::JIT_BACKGROUND_SLICE,
            jit_automatic_nbit_address_space: false,
            jit_ignore_instruction_limit: false,
            jit_cache: true,
            #[cfg(feature = "jit")]
            jit_cache_output: None,
            aot: true,
            #[cfg(feature = "aot")]
            aot_output: None,
            default_exit_function: None,
            use_shared_execute_segments: true,
            use_shared_rodata: true,
            ebreak_locations: Vec::new(),
            libc_fastpath: false,
        }
    }
}
