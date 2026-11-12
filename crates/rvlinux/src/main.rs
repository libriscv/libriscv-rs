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
    fuel: Option<u64>,
    memory_mib: Option<u64>,
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
    if cfg!(feature = "jit") {
        println!("  -j, --jit          Compile to native code with the Cranelift JIT");
        println!("  -B, --background   Compile in the background while interpreting (default)");
        println!("      --no-background Compile each segment before running it");
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
            "-j" | "--jit" if cfg!(feature = "jit") => args.jit = true,
            "-B" | "--background" if cfg!(feature = "jit") => args.no_background = false,
            "--no-background" if cfg!(feature = "jit") => args.no_background = true,
            "-f" | "--fuel" => args.fuel = Some(num(&mut it, "--fuel")?),
            "-m" | "--memory" => args.memory_mib = Some(num(&mut it, "--memory")?),
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
    let binary = match std::fs::read(&args.program) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Could not read {}: {e}", args.program.display());
            return ExitCode::FAILURE;
        }
    };
    let options = libriscv::MachineOptions {
        memory_max: args.memory_mib.map_or(DEFAULT_MEMORY, |m| m << 20),
        verbose_loader: args.verbose,
        jit: args.jit,
        // C++ rvlinux compiles in the background by default too.
        jit_background: !args.no_background,
        ..Default::default()
    };
    let mut machine = match libriscv::Machine::new(&binary, options) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Exception: {e}");
            return ExitCode::FAILURE;
        }
    };
    // TODO(phase 2): setup_linux (argv/env/auxv on the stack) and the Linux
    // system calls. Until then only freestanding guests run: write, exit,
    // exit_group, brk and stubs.
    let _ = &args.guest_args;
    machine.setup_minimal_syscalls();
    // C++ rvlinux installs `setup_posix_threads`, whose exit/exit_group
    // replace the minimal ones: for the main thread they stop and then
    // `set_result(uint32_t status)`, which sign-extends the low 32 bits of
    // a0 into the reported exit code. TODO(phase 5): threads.
    let exit = |m: &mut libriscv::Machine| {
        let status = m.return_value() as u32;
        m.stop();
        m.cpu
            .set_reg(libriscv::registers::REG_ARG0, status as i32 as u64);
        Ok(())
    };
    machine.install_syscall_handler(93, exit);
    machine.install_syscall_handler(94, exit);

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
    let _ = std::io::Write::flush(&mut std::io::stdout());
    ExitCode::SUCCESS
}
