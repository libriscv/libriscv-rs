//! Machine construction options — port of C++ `MachineOptions<W>` (common.hpp).
//!
//! Only interpreter-relevant options are carried over. Binary-translation,
//! libtcc and asmjit options are intentionally absent (see PORTING.md, "Scope").
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
        }
    }
}
