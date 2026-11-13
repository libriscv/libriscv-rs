//! `rvlinux` — run a RISC-V Linux userspace ELF under libriscv.
//!
//! Port of `emulator/src/main.cpp`. Output format of the completion summary
//! must stay byte-compatible with the C++ emulator (see `testutil::parse_run_output`)
//! so the two can be diffed directly.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Default)]
struct CliArgs {
    verbose: bool,
    accurate: bool,
    silent: bool,
    jit: bool,
    no_background: bool,
    nbit: bool,
    jit_save_object: Option<PathBuf>,
    jit_save_shared: Option<PathBuf>,
    jit_load: Vec<PathBuf>,
    no_jit_cache: bool,
    quit: bool,
    libc_fastpath: bool,
    sandbox: bool,
    proxy: bool,
    allowed_files: Vec<String>,
    fuel: Option<u64>,
    memory_mib: Option<u64>,
    call_function: Option<String>,
    program: PathBuf,
    guest_args: Vec<String>,
}

fn usage(name: &str) {
    println!("Usage: {name} [options] <program> [args]");
    println!("Options:");
    println!("  -h, --help         Print this help message");
    println!("  -v, --verbose      Enable verbose loader output");
    println!("  -a, --accurate     Accurate instruction counting");
    println!("  -s, --silent       Suppress program completion information");
    println!("  -f, --fuel amt     Set max instructions until program halts");
    println!("  -m, --memory amt   Set max memory size in MiB (default: 4000 MiB)");
    println!("  -n, --no-translate Accepted for compatibility (always interpreted)");
    println!("  -S, --sandbox      Enable strict sandbox");
    println!(
        "  -P, --proxy        Enable proxy mode, allowing access to all files (disabling the sandbox)"
    );
    println!("  -A, --allow file   Allow file to be opened by the guest");
    println!("  -c, --call func    Call a function after loading the program");
    println!("  -Q, --quit         Quit after loading the program (to save JIT code)");
    println!(
        "      --libc-fastpath  Hot-patch memcpy, memset, strlen etc. with native implementations"
    );
    if cfg!(feature = "jit") {
        println!("  -j, --jit          Compile to native code with the Cranelift JIT");
        println!("  -B, --background   Compile in the background while interpreting (default)");
        println!("      --no-background Compile each segment before running it");
        println!(
            "      --nbit-address-space  Mask arena addresses in translated code instead of bounds-checking them"
        );
        println!(
            "JIT code cache (Cranelift output saved for later runs; implies --jit --no-background):"
        );
        println!(
            "      --jit-save-object file  Save the compiled code as an object file to link into a program"
        );
        println!(
            "      --jit-save-shared file  Save the compiled code as a shared object (links with $CC or cc)"
        );
        println!("      --jit-load file    Load saved code from a shared object (repeatable)");
        println!(
            "      --no-jit-cache     Ignore saved code (loaded, or linked into this program)"
        );
    }
}

fn parse_args() -> Result<CliArgs, String> {
    let mut it = std::env::args();
    let name = it.next().unwrap_or_else(|| "rvlinux".into());
    let mut args = CliArgs::default();
    let num = |it: &mut std::env::Args, flag: &str| -> Result<u64, String> {
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        v.parse().map_err(|_| format!("Invalid number: {v}"))
    };
    let path = |it: &mut std::env::Args, flag: &str| -> Result<PathBuf, String> {
        it.next()
            .map(PathBuf::from)
            .ok_or_else(|| format!("{flag} requires a value"))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                usage(&name);
                std::process::exit(0);
            }
            "-v" | "--verbose" => args.verbose = true,
            "-a" | "--accurate" => args.accurate = true,
            "-s" | "--silent" => args.silent = true,
            "-n" | "--no-translate" => {}
            "-S" | "--sandbox" => args.sandbox = true,
            "-P" | "--proxy" => args.proxy = true,
            "-A" | "--allow" => args.allowed_files.push(
                it.next()
                    .ok_or_else(|| "--allow requires a value".to_owned())?,
            ),
            "-j" | "--jit" if cfg!(feature = "jit") => args.jit = true,
            "-B" | "--background" if cfg!(feature = "jit") => args.no_background = false,
            "--no-background" if cfg!(feature = "jit") => args.no_background = true,
            "--nbit-address-space" if cfg!(feature = "jit") => args.nbit = true,
            "--jit-save-object" if cfg!(feature = "jit") => {
                args.jit_save_object = Some(path(&mut it, "--jit-save-object")?);
            }
            "--jit-save-shared" if cfg!(feature = "jit") => {
                args.jit_save_shared = Some(path(&mut it, "--jit-save-shared")?);
            }
            "--jit-load" if cfg!(feature = "jit") => {
                args.jit_load.push(path(&mut it, "--jit-load")?);
            }
            "--no-jit-cache" if cfg!(feature = "jit") => args.no_jit_cache = true,
            "-Q" | "--quit" => args.quit = true,
            "--libc-fastpath" => args.libc_fastpath = true,
            "-f" | "--fuel" => args.fuel = Some(num(&mut it, "--fuel")?),
            "-m" | "--memory" => args.memory_mib = Some(num(&mut it, "--memory")?),
            "-c" | "--call" => {
                let func = it
                    .next()
                    .ok_or_else(|| "--call requires a value".to_owned())?;
                if args.verbose {
                    println!("* Function to VMCall: {func}");
                }
                args.call_function = Some(func);
            }
            s if s.starts_with('-') => return Err(format!("Unknown option: {s}")),
            _ => {
                args.program = PathBuf::from(arg);
                args.guest_args = it.collect();
                break;
            }
        }
    }
    if args.program.as_os_str().is_empty() {
        usage(&name);
        return Err("No program given".into());
    }
    if args.jit_save_object.is_some() || args.jit_save_shared.is_some() {
        // Saving needs every segment compiled by the time the program ends.
        args.jit = true;
        args.no_background = true;
    }
    Ok(args)
}

/// C++ `MAX_MEMORY` without an encompassing arena.
const DEFAULT_MEMORY: u64 = 4000 << 20;
/// C++ default fuel: time out after ~30bn instructions.
const DEFAULT_FUEL: u64 = 30_000_000_000;

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut binary = match std::fs::read(&args.program) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Could not read {}: {e}", args.program.display());
            return ExitCode::FAILURE;
        }
    };
    // The guest's argv: the program and its arguments.
    let mut guest_args = vec![args.program.to_string_lossy().into_owned()];
    guest_args.extend(args.guest_args.iter().cloned());
    let mut is_dynamic = false;
    if let Some(interp) = libriscv::elf_interpreter(&binary) {
        // A dynamic executable: run its dynamic linker instead, which maps
        // the program and its libraries at runtime, with the program as
        // its first argument (C++ `main`).
        let interp_path = interp.trim_end_matches('\0');
        binary = match std::fs::read(interp_path).or_else(|_| std::fs::read(DYNAMIC_LINKER)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Could not read the dynamic linker {interp_path}: {e}");
                return ExitCode::FAILURE;
            }
        };
        // C++ keeps the NUL terminator of PT_INTERP in argv[0], so its
        // string takes one byte more on the initial stack.
        guest_args.insert(0, interp);
        is_dynamic = true;
    }
    #[cfg(feature = "jit")]
    for file in &args.jit_load {
        match libriscv::jit_cache::load_shared_object(file) {
            Ok(n) if args.verbose => {
                println!(
                    "* Loaded saved JIT code for {n} segments from {}",
                    file.display()
                );
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Could not load {}: {e}", file.display());
                return ExitCode::FAILURE;
            }
        }
    }
    #[cfg(feature = "jit")]
    let jit_cache_output = (args.jit_save_object.is_some() || args.jit_save_shared.is_some())
        .then(libriscv::jit_cache::JitCacheOutput::new);
    let options = libriscv::MachineOptions {
        memory_max: args.memory_mib.map_or(DEFAULT_MEMORY, |m| m << 20),
        verbose_loader: args.verbose,
        jit: args.jit,
        // C++ rvlinux compiles in the background by default too.
        jit_background: !args.no_background,
        jit_automatic_nbit_address_space: args.nbit,
        jit_cache: !args.no_jit_cache,
        #[cfg(feature = "jit")]
        jit_cache_output: jit_cache_output.clone(),
        libc_fastpath: args.libc_fastpath,
        ..Default::default()
    };
    let mut machine = match libriscv::Machine::new(&binary, options) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Exception: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = setup_linux(&mut machine, &args, &guest_args, is_dynamic) {
        eprintln!("Exception: {e}");
        return ExitCode::FAILURE;
    }
    #[cfg(feature = "jit")]
    let save = || save_jit_code(&args, jit_cache_output.as_ref());
    #[cfg(not(feature = "jit"))]
    let save = || ExitCode::SUCCESS;
    if args.quit {
        return save();
    }

    // `--fuel 0` means no limit, as in C++.
    let fuel = match args.fuel {
        Some(0) => u64::MAX,
        Some(f) => f,
        None => DEFAULT_FUEL,
    };
    let t0 = std::time::Instant::now();
    let result = if args.accurate {
        machine.simulate(fuel)
    } else {
        let pc = machine.cpu.pc();
        machine.simulate_inaccurate(pc)
    };
    if let Err(e) = result {
        // Same line as the C++ emulator (which also prints the current
        // instruction, the registers and a backtrace around it).
        println!(
            ">>> Machine exception {}: {} (data: 0x{:X})",
            e.kind as u8, e.message, e.data
        );
        if matches!(
            e.kind,
            libriscv::ExceptionKind::UnimplementedInstruction
                | libriscv::ExceptionKind::MisalignedInstruction
        ) {
            println!(">>> Is an instruction extension disabled?");
            println!(
                ">>> A-extension: {}  C-extension: {}  V-extension: {}",
                u8::from(libriscv::ATOMICS_ENABLED),
                u8::from(libriscv::COMPRESSED_ENABLED),
                libriscv::VECTOR_EXTENSION
            );
        }
        run_sighandler(&mut machine, signal_for_exception(e.kind));
    }
    let runtime = t0.elapsed().as_secs_f64();

    if !args.silent {
        // Byte-compatible with the C++ emulator; testutil::parse_run_output
        // depends on it.
        let retval = machine.return_value();
        println!(
            ">>> Program exited, exit code = {} (0x{retval:X})",
            retval as i64
        );
        if args.accurate {
            let n = machine.instruction_counter();
            println!(
                "Instructions executed: {n}  Runtime: {:.3}ms  Insn/s: {:.0}mi/s",
                runtime * 1000.0,
                n as f64 / (runtime * 1e6)
            );
        } else {
            println!(
                "Runtime: {:.3}ms   (Use --accurate for instruction counting)",
                runtime * 1000.0
            );
        }
    }
    if let Some(func) = &args.call_function {
        // C++ also falls back to asking the guest's symbol function (system
        // call 500); not ported, as that needs userdata.
        let addr = machine.address_of(func).unwrap_or(0);
        if addr != 0 {
            println!("Calling function {func} @ 0x{addr:X}");
            if let Err(e) = machine.vmcall(addr, ()) {
                // An uncaught exception terminates the C++ emulator.
                let _ = std::io::Write::flush(&mut std::io::stdout());
                eprintln!("Exception: {e}");
                return ExitCode::FAILURE;
            }
        } else {
            println!("Error: Function {func} not found, not able to call");
        }
    }
    let _ = std::io::Write::flush(&mut std::io::stdout());
    save()
}

/// Write the code the JIT compiled to the `--jit-save-*` files.
#[cfg(feature = "jit")]
fn save_jit_code(args: &CliArgs, output: Option<&libriscv::jit_cache::JitCacheOutput>) -> ExitCode {
    let Some(output) = output else {
        return ExitCode::SUCCESS;
    };
    let mut status = ExitCode::SUCCESS;
    for (file, shared) in [
        (&args.jit_save_object, false),
        (&args.jit_save_shared, true),
    ] {
        let Some(file) = file else { continue };
        let saved = if shared {
            output.write_shared_object(file)
        } else {
            output.write_object(file)
        };
        match saved {
            Ok(()) if args.verbose => {
                println!(
                    "* Saved JIT code for {} segments to {}",
                    output.len(),
                    file.display()
                );
            }
            Ok(()) => {}
            Err(e) => {
                eprintln!("Could not save JIT code to {}: {e}", file.display());
                status = ExitCode::FAILURE;
            }
        }
    }
    status
}

/// The guest environment (as in C++).
const GUEST_ENV: [&str; 3] = ["LC_CTYPE=C", "LC_ALL=C", "RUST_BACKTRACE=full"];

/// The dynamic linker used when the interpreter named by a dynamic
/// executable can't be read (C++ `DYNAMIC_LINKER`).
const DYNAMIC_LINKER: &str = "/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1";

/// Libraries the guest may open, from `/lib/riscv64-linux-gnu/` (C++
/// `filter_open`, for dynamic executables).
const GUEST_LIBS: [&str; 9] = [
    "libdl.so.2",
    "libm.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    "libatomic.so.1",
    "libstdc++.so.6",
    "libresolv.so.2",
    "libnss_dns.so.2",
    "libnss_files.so.2",
];
const SANDBOX_LIBDIR: &str = "/lib/riscv64-linux-gnu/";
const REAL_LIBDIR: &str = "/usr/riscv64-linux-gnu/lib/";

/// The Linux environment of the C++ emulator (`full_linux_guest`): initial
/// stack with argv/env/auxv, Linux system calls with a sandboxed file
/// system, and POSIX threads.
fn setup_linux(
    machine: &mut libriscv::Machine,
    args: &CliArgs,
    guest_args: &[String],
    is_dynamic: bool,
) -> libriscv::Result<()> {
    machine.setup_linux(guest_args, &GUEST_ENV)?;
    machine.setup_linux_syscalls(true, true);

    let program = guest_args.first().cloned().unwrap_or_default();
    // The dynamic linker may open the program it runs.
    let dynamic_program = guest_args.get(1).filter(|_| is_dynamic).cloned();
    let verbose = args.verbose;
    let proxy = args.proxy;
    let allowed = args.allowed_files.clone();
    if let Some(fds) = machine.fds_mut() {
        fds.permit_filesystem = !args.sandbox;
        fds.permit_sockets = !args.sandbox;
        if args.proxy {
            if args.sandbox {
                eprintln!("Warning: Proxy mode is enabled, but sandbox is also enabled");
            }
            fds.permit_filesystem = true;
            fds.permit_sockets = true;
            fds.proxy_mode = true;
            if let Ok(cwd) = std::env::current_dir() {
                fds.cwd = cwd.to_string_lossy().into_owned();
            }
        }
        let cwd = fds.cwd.clone();
        fds.filter_readlink = Some(Box::new(move |path: &mut String| {
            if path == "/proc/self/exe" {
                *path = format!("{cwd}/program");
                return true;
            }
            if path == "/proc/self/fd/1" || path == "/proc/self/fd/2" {
                return true;
            }
            eprintln!("Guest wanted to readlink: {path} (denied)");
            false
        }));
        fds.filter_open = Some(Box::new(move |path: &mut String| {
            const ALLOWED: [&str; 7] = [
                "/etc/hostname",
                "/etc/hosts",
                "/etc/nsswitch.conf",
                "/etc/host.conf",
                "/etc/resolv.conf",
                "/dev/urandom",
                "/etc/ssl/certs/ca-certificates.crt",
            ];
            if ALLOWED.contains(&path.as_str()) {
                return true;
            }
            if path == "/program" {
                // The fake program path opens the real program
                *path = program.clone();
                return true;
            }
            if let Some(lib) = path.strip_prefix(SANDBOX_LIBDIR) {
                if !GUEST_LIBS.contains(&lib) {
                    if verbose {
                        eprintln!("Guest wanted to open: {path} (denied)");
                    }
                    return false;
                }
                if verbose {
                    eprintln!("Guest wanted to open: {path} (allowed)");
                }
                *path = format!("{REAL_LIBDIR}{lib}");
                return true;
            }
            if dynamic_program.as_deref() == Some(path.as_str()) {
                return true;
            }
            if proxy || allowed.iter().any(|a| a == path) {
                return true;
            }
            if verbose {
                eprintln!("Guest wanted to open: {path} (denied)");
            }
            false
        }));
    }
    if args.proxy {
        machine.set_stdin(|_, buf| {
            use std::io::Read;
            std::io::stdin()
                .read(buf)
                .map_or(-1, |n| i64::try_from(n).unwrap_or(i64::MAX))
        });
    }
    machine.setup_posix_threads();
    Ok(())
}

/// The signal a machine exception corresponds to (C++ `signal_for_exception`).
const fn signal_for_exception(kind: libriscv::ExceptionKind) -> i32 {
    use libriscv::ExceptionKind as K;
    match kind {
        K::IllegalOpcode
        | K::IllegalOperation
        | K::UnimplementedInstruction
        | K::UnimplementedInstructionLength
        | K::MisalignedInstruction
        | K::InvalidAlignment => 4, // SIGILL
        _ => 11, // SIGSEGV
    }
}

/// After a fault, give the guest's signal handler (or its SIGSEGV
/// handler) a chance to run, for at most 60k instructions, as C++
/// `run_sighandler` does. Its errors are ignored.
fn run_sighandler(machine: &mut libriscv::Machine, signal: i32) {
    use libriscv::registers::{REG_ARG0, REG_RA};
    const SIGSEGV: i32 = 11;
    let unset =
        |m: &mut libriscv::Machine, sig: i32| m.sigaction(sig).map_or(true, |a| a.is_unset());
    let mut signal = signal;
    if unset(machine, signal) {
        if signal == SIGSEGV || unset(machine, SIGSEGV) {
            return;
        }
        let segv = machine.sigaction(SIGSEGV).map(|a| *a);
        if let (Ok(segv), Ok(action)) = (segv, machine.sigaction(signal)) {
            *action = segv;
        }
        signal = SIGSEGV;
    }
    let Ok(action) = machine.sigaction(signal) else {
        return;
    };
    let handler = action.handler;
    // Avoid re-triggering
    action.handler = 0;
    let ra = machine.cpu.reg(REG_RA);
    if machine.stack_push(&ra.to_le_bytes()).is_ok() {
        let pc = machine.cpu.pc();
        machine.cpu.set_reg(REG_RA, pc);
        machine.cpu.set_reg(REG_ARG0, signal as u64);
        if machine.cpu.jump(handler).is_ok() {
            let _ = machine.simulate(60_000);
        }
    }
    if let Ok(action) = machine.sigaction(signal) {
        action.handler = handler;
    }
}
