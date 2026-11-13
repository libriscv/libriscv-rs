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
    /// With `jit`, mask guest addresses with `arena size - 1` in native code
    /// instead of bounds-checking them (C++
    /// `translate_automatic_nbit_address_space`). Much cheaper loads and
    /// stores, but native code then reads and writes anywhere in the arena,
    /// read-only data and the first page included, and accesses outside it
    /// wrap around instead of going to the page table. Only power-of-two
    /// arenas qualify; otherwise (reported with `verbose_loader`) the
    /// accesses stay checked. The interpreter is unaffected.
    pub jit_automatic_nbit_address_space: bool,
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
            default_exit_function: None,
            use_shared_execute_segments: true,
        }
    }
}
