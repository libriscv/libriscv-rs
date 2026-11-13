//! VM calls and preemption: port of `tests/unit/vmcall.cpp`, plus the
//! clobbering/non-clobbering preemption variants, error handling,
//! `default_exit_function` and nested runs under background JIT
//! compilation.
//!

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
// Every test runs a glibc guest, and the distribution's glibc is RV64GC.
#![cfg(all(feature = "ext-a", feature = "ext-c", feature = "ext-f"))]

use std::sync::atomic::{AtomicBool, Ordering};

use libriscv::registers::{REG_ARG0, REG_RA, REG_SP};
use libriscv::{ExceptionKind, GuestPod, Machine, MachineOptions, OnStack};

const MAX_MEMORY: u64 = 8 << 20;
const MAX_INSTRUCTIONS: u64 = 10_000_000;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];

fn build(code: &str) -> Vec<u8> {
    testutil::build_and_load(code, "-O2 -static").unwrap_or_else(|e| panic!("{e}"))
}

fn machine_with(binary: &[u8], options: MachineOptions) -> Machine {
    let mut machine = Machine::new(
        binary,
        MachineOptions {
            memory_max: MAX_MEMORY,
            ..options
        },
    )
    .unwrap();
    // We need to install Linux system calls for maximum gucciness, and a
    // Linux environment for runtimes to work well.
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["vmcall"], &ENV).unwrap();
    machine
}

fn machine(binary: &[u8]) -> Machine {
    machine_with(binary, MachineOptions::default())
}

/// The C++ tests' `State`, as userdata. The printer only gets `&Machine`,
/// hence the atomic.
#[derive(Default)]
struct State {
    output_is_hello_world: AtomicBool,
}

/// C++: `state->output_is_hello_world = (text == "Hello World!")`.
fn hello_printer(m: &Machine, data: &[u8]) {
    let state = m.userdata::<State>().unwrap();
    state
        .output_is_hello_world
        .store(data == b"Hello World!", Ordering::Relaxed);
}

/// C++ `machine.set_userdata(&state); machine.set_printer(...)`.
fn catch_hello(m: &mut Machine) {
    m.set_userdata(State::default());
    m.set_printer(hello_printer);
}

fn reset_hello(m: &Machine) {
    let state = m.userdata::<State>().unwrap();
    state.output_is_hello_world.store(false, Ordering::Relaxed);
}

fn hello_world(m: &Machine) -> bool {
    let state = m.userdata::<State>().unwrap();
    state.output_is_hello_world.load(Ordering::Relaxed)
}

#[test]
fn vm_function_call() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	extern long write(int, const void*, unsigned long);
	__attribute__((used, retain))
	void hello() {
		write(1, "Hello World!", 12);
	}

	int main() {
		return 666;
	}"#,
    );
    let mut machine = machine(&binary);
    catch_hello(&mut machine);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    assert!(!hello_world(&machine));

    let hello_address = machine.address_of("hello").unwrap();
    assert_ne!(hello_address, 0);

    machine.vmcall(hello_address, ()).unwrap();
    assert!(hello_world(&machine));
}

#[test]
fn vm_call_return_values() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	__attribute__((used, retain))
	const char* hello() {
		return "Hello World!";
	}

	static struct Data {
		int val1;
		int val2;
		float f1;
	} data = {.val1 = 1, .val2 = 2, .f1 = 3.0f};
	__attribute__((used, retain))
	extern struct Data* structs() {
		return &data;
	}

	int main() {
		return 666;
	}"#,
    );
    let mut machine = machine(&binary);

    let hello_address = machine.address_of("hello").unwrap();
    assert_ne!(hello_address, 0);
    // Returning a string
    let s = machine.vmcall(hello_address, ()).unwrap();
    assert_eq!(machine.memory.memstring(s).unwrap(), "Hello World!");

    let structs_address = machine.address_of("structs").unwrap();
    assert_ne!(structs_address, 0);
    // Returning a pointer to a structure
    let p = machine.vmcall(structs_address, ()).unwrap();
    let mut data = [0u8; 12];
    machine.copy_from_guest(&mut data, p).unwrap();
    assert_eq!(i32::from_le_bytes(data[0..4].try_into().unwrap()), 1);
    assert_eq!(i32::from_le_bytes(data[4..8].try_into().unwrap()), 2);
    assert_eq!(f32::from_le_bytes(data[8..12].try_into().unwrap()), 3.0);
}

#[test]
fn vm_call_enum_values() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	#include <assert.h>
	int do_syscall(int value) {
		register long         a0 __asm__("a0") = value;
		register long syscall_id __asm__("a7") = 0;

		__asm__ volatile ("ecall" : "+r"(a0) : "r"(syscall_id));
		return a0;
	}
	__attribute__((used, retain))
	int mycall(int value) {
		assert(value == 1);
		return do_syscall(value);
	}

	int main() {
		return 666;
	}"#,
    );
    let mut machine = machine(&binary);

    #[derive(Clone, Copy)]
    #[repr(i32)]
    enum MyEnum {
        Hello = 1,
        World = 2,
    }
    // Enums pass as their integer value (C++ `int(args)`).
    impl libriscv::VmCallArg for MyEnum {
        fn pass(&self, m: &mut Machine, regs: &mut libriscv::ArgRegs) -> libriscv::Result<()> {
            (*self as i32).pass(m, regs)
        }
    }

    machine.install_syscall_handler(0, |m| {
        assert_eq!(m.sysarg(0) as i32, MyEnum::Hello as i32);
        m.set_result(MyEnum::World as u64);
        Ok(())
    });
    let r = machine.vmcall("mycall", MyEnum::Hello).unwrap();
    assert_eq!(r as i32, MyEnum::World as i32);
}

/// The guest functions of C++ "VM function call in fork", called on the
/// main machine.
#[test]
fn argument_types() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	#include <assert.h>
	#include <string.h>

	__attribute__((used, retain))
	int str(const char *arg) {
		assert(strcmp(arg, "Hello") == 0);
		return 1;
	}

	struct Data {
		int val1;
		int val2;
		float f1;
	};
	__attribute__((used, retain))
	int structs(struct Data *data) {
		assert(data->val1 == 1);
		assert(data->val2 == 2);
		assert(data->f1 == 3.0f);
		return 2;
	}

	__attribute__((used, retain))
	int ints(long i1, long i2, long i3) {
		assert(i1 == 123);
		assert(i2 == 456);
		assert(i3 == 456);
		return 3;
	}

	__attribute__((used, retain))
	int fps(float f1, double d1) {
		assert(f1 == 1.0f);
		assert(d1 == 2.0);
		return 4;
	}

	__attribute__((used, retain))
	long mixed(int a, double b, const char* s, float c, long d) {
		assert(a == -1);
		assert(b == 0.5);
		assert(strcmp(s, "mixed") == 0);
		assert(c == -2.0f);
		assert(d == 0x123456789);
		return 5;
	}

	int main() {
		return 666;
	}"#,
    );
    let mut m = machine(&binary);
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(m.return_value() as i32, 666);

    assert_eq!(m.vmcall("str", "Hello").unwrap(), 1);
    assert_eq!(m.vmcall("str", String::from("Hello")).unwrap(), 1);
    let hello = String::from("Hello");
    assert_eq!(m.vmcall("str", &hello).unwrap(), 1);

    assert_eq!(m.vmcall("structs", OnStack(Data(1, 2, 3.0))).unwrap(), 2);

    let intval: i64 = 456;
    assert_eq!(m.vmcall("ints", (123i64, intval, &intval)).unwrap(), 3);
    assert_eq!(m.vmcall("fps", (1.0f32, 2.0f64)).unwrap(), 4);
    assert_eq!(
        m.vmcall("mixed", (-1, 0.5, "mixed", -2.0f32, 0x1_2345_6789u64))
            .unwrap(),
        5
    );
}

#[test]
fn vm_function_call_in_fork() {
    // The global variable 'value' should get forked as value=1. We assert
    // this, then we set value=0. New forks should continue to see value=1
    // as they are forked from the main VM where value is still 1.
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	#include <assert.h>
	#include <string.h>
	extern long write(int, const void*, unsigned long);
	static int value = 0;

	__attribute__((used, retain))
	void hello() {
		assert(value == 1);
		value = 0;
		write(1, "Hello World!", 12);
	}

	__attribute__((used, retain))
	int str(const char *arg) {
		assert(strcmp(arg, "Hello") == 0);
		return 1;
	}

	struct Data {
		int val1;
		int val2;
		float f1;
	};
	__attribute__((used, retain))
	int structs(struct Data *data) {
		assert(data->val1 == 1);
		assert(data->val2 == 2);
		assert(data->f1 == 3.0f);
		return 2;
	}

	__attribute__((used, retain))
	int ints(long i1, long i2, long i3) {
		assert(i1 == 123);
		assert(i2 == 456);
		assert(i3 == 456);
		return 3;
	}

	__attribute__((used, retain))
	int fps(float f1, double d1) {
		assert(f1 == 1.0f);
		assert(d1 == 2.0);
		return 4;
	}

	int main() {
		value = 1;
		return 666;
	}"#,
    );
    // The forking constructor does not support a flat arena.
    let mut machine = machine_with(
        &binary,
        MachineOptions {
            use_memory_arena: false,
            ..Default::default()
        },
    );
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);

    // Test many forks
    for _ in 0..10 {
        let mut fork = machine.fork().unwrap();
        assert!(!fork.memory.uses_flat_memory_arena());

        catch_hello(&mut fork);

        let hello_address = fork.address_of("hello").unwrap();
        assert_ne!(hello_address, 0);

        // Execute guest function
        fork.vmcall(hello_address, ()).unwrap();
        assert!(hello_world(&fork));

        assert_eq!(fork.vmcall("str", "Hello").unwrap(), 1);
        assert_eq!(fork.vmcall("str", String::from("Hello")).unwrap(), 1);
        let hello = String::from("Hello");
        assert_eq!(fork.vmcall("str", &hello).unwrap(), 1);

        assert_eq!(fork.vmcall("structs", OnStack(Data(1, 2, 3.0))).unwrap(), 2);

        let intval: i64 = 456;
        assert_eq!(fork.vmcall("ints", (123i64, intval, &intval)).unwrap(), 3);
        assert_eq!(fork.vmcall("fps", (1.0f32, 2.0f64)).unwrap(), 4);
    }
}

/// C++ `struct Data { int val1; int val2; float f1; }`.
#[derive(Clone, Copy)]
struct Data(i32, i32, f32);

impl GuestPod for Data {
    const SIZE: usize = 12;
    const ALIGN: usize = 4;
    fn write_guest(&self, out: &mut Vec<u8>) {
        self.0.write_guest(out);
        self.1.write_guest(out);
        self.2.write_guest(out);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Data(
            i32::read_guest(bytes),
            i32::read_guest(&bytes[4..]),
            f32::read_guest(&bytes[8..]),
        )
    }
}

const PREEMPT_PROGRAM: &str = r#"
	extern long write(int, const void*, unsigned long);
	long syscall1(long n, long arg0) {
		register long a0 __asm__("a0") = arg0;
		register long syscall_id __asm__("a7") = n;

		__asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));

		return a0;
	}

	__attribute__((used, retain))
	long start() {
		syscall1(500, 1234567);
		return 1;
	}
	__attribute__((used, retain))
	void preempt(int arg) {
		write(1, "Hello World!", arg);
	}

	int main() {
		syscall1(500, 1234567);
		return 666;
	}"#;

fn preempt_handler(m: &mut Machine) -> libriscv::Result<()> {
    assert_eq!(m.sysarg(0) as i32, 1_234_567);
    let func = m.address_of("preempt").unwrap();
    assert_ne!(func, 0);
    m.preempt(15_000, func, "Hello World!".len())?;
    Ok(())
}

fn vm_call_and_preemption_in(mut machine: Machine) {
    catch_hello(&mut machine);
    machine.install_syscall_handler(500, preempt_handler);

    assert!(!hello_world(&machine));
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert!(hello_world(&machine));
    assert_eq!(machine.return_value() as i32, 666);

    for _ in 0..10 {
        reset_hello(&machine);
        let func = machine.address_of("start").unwrap();
        assert_ne!(func, 0);
        let r = machine.vmcall_limited(15_000, func, ()).unwrap();
        assert_eq!(r as i32, 1);
        assert!(hello_world(&machine));
    }
}

#[test]
fn vm_call_and_preemption() {
    if !testutil::need_compiler() {
        return;
    }
    vm_call_and_preemption_in(machine(&build(PREEMPT_PROGRAM)));
}

#[test]
fn vm_call_and_stop_instruction() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	extern long write(int, const void*, unsigned long);
	long syscall1(long n, long arg0) {
		register long a0 __asm__("a0") = arg0;
		register long syscall_id __asm__("a7") = n;

		__asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));

		return a0;
	}
	void return_fast1(long retval)
	{
		register long a0 __asm__("a0") = retval;

		__asm__ volatile (".insn i SYSTEM, 0, x0, x0, 0x7ff" :: "r"(a0));
		__builtin_unreachable();
	}

	__attribute__((used, retain))
	long start() {
		syscall1(500, 1234567);
		return_fast1(1234);
		return 5678;
	}
	__attribute__((used, retain))
	long preempt(int arg) {
		write(1, "Hello World!", arg);
		return_fast1(777);
	}

	int main() {
		syscall1(500, 1234567);
		return_fast1(777);
		return 666;
	}"#,
    );
    let mut machine = machine(&binary);
    catch_hello(&mut machine);
    machine.install_syscall_handler(500, |m| {
        assert_eq!(m.sysarg(0) as i32, 1_234_567);
        let func = m.address_of("preempt").unwrap();
        assert_ne!(func, 0);
        let result = m.preempt(15_000, func, "Hello World!".len())?;
        assert_eq!(result, 777);
        Ok(())
    });

    assert!(!hello_world(&machine));
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert!(hello_world(&machine));
    assert_eq!(machine.return_value() as i32, 777);

    for _ in 0..10 {
        reset_hello(&machine);
        let func = machine.address_of("start").unwrap();
        assert_ne!(func, 0);
        machine.vmcall_limited(15_000, func, ()).unwrap();
        assert_eq!(machine.return_value() as i32, 1234);
        assert!(hello_world(&machine));
    }
}

#[test]
fn vm_call_with_arrays_and_vectors() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	__attribute__((used, retain))
	int pass_iarray(const int* data, unsigned size) {
		if (size != 3)
			return 0;
		if (data[0] != 1 || data[1] != 2 || data[2] != 3)
			return 0;
		return 1;
	}

	__attribute__((used, retain))
	int pass_farray(const float* data, unsigned size) {
		if (size != 3)
			return 0;
		if (data[0] != 1.0f || data[1] != 2.0f || data[2] != 3.0f)
			return 0;
		return 1;
	}

	struct Data {
		int val1;
		int val2;
		float f1;
	};
	__attribute__((used, retain))
	int pass_struct(const struct Data* data, unsigned size) {
		if (size != 3)
			return 0;
		if (data[0].val1 != 1 || data[0].val2 != 2 || data[0].f1 != 3.0f)
			return 0;
		if (data[1].val1 != 4 || data[1].val2 != 5 || data[1].f1 != 6.0f)
			return 0;
		if (data[2].val1 != 7 || data[2].val2 != 8 || data[2].f1 != 9.0f)
			return 0;
		return 1;
	}

	int main() {
		return 666;
	}"#,
    );
    let mut machine = machine(&binary);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);

    // The array is pushed on the stack, so it becomes a pointer argument.
    let iarray: [i32; 3] = [1, 2, 3];
    let r = machine
        .vmcall("pass_iarray", (iarray, iarray.len()))
        .unwrap();
    assert_eq!(r as i32, 1);
    // A reference to an array works too.
    let array_ref = &iarray;
    let r = machine
        .vmcall("pass_iarray", (array_ref, array_ref.len()))
        .unwrap();
    assert_eq!(r as i32, 1);

    let farray: [f32; 3] = [1.0, 2.0, 3.0];
    let r = machine
        .vmcall("pass_farray", (farray, farray.len()))
        .unwrap();
    assert_eq!(r as i32, 1);

    let vec = vec![Data(1, 2, 3.0), Data(4, 5, 6.0), Data(7, 8, 9.0)];
    let r = machine.vmcall("pass_struct", (&vec, vec.len())).unwrap();
    assert_eq!(r as i32, 1);
    let slice: &[Data] = &vec;
    let r = machine.vmcall("pass_struct", (slice, slice.len())).unwrap();
    assert_eq!(r as i32, 1);
}

const REGS_PROGRAM: &str = r#"
	static long counter = 0;
	__attribute__((used, retain))
	long bump(long n) {
		counter += n;
		return counter;
	}
	__attribute__((used, retain))
	long spin(long n) {
		for (volatile long i = 0; i < n; i++);
		return n;
	}
	__attribute__((used, retain))
	long crash() {
		*(volatile long*)0x8 = 1;
		return 0;
	}
	int main() {
		return 0;
	}"#;

/// `preempt` restores every register, PC included; `preempt_clobbering`
/// leaves what the function left.
#[test]
fn preempt_restores_or_clobbers_registers() {
    if !testutil::need_compiler() {
        return;
    }
    let mut m = machine(&build(REGS_PROGRAM));
    m.simulate(MAX_INSTRUCTIONS).unwrap();

    let before = m.cpu.registers().clone();
    let counter = m.instruction_counter();
    assert_eq!(m.preempt(10_000, "bump", 5).unwrap(), 5);
    assert_eq!(m.cpu.registers(), &before);
    // The function's instructions were counted.
    assert!(m.instruction_counter() > counter);

    assert_eq!(m.preempt_clobbering(10_000, "bump", 5).unwrap(), 10);
    assert_ne!(m.cpu.registers(), &before);
    assert_eq!(m.cpu.reg(REG_ARG0), 10);
    // Called with a 16-byte red zone below the old stack pointer, returned
    // through the exit function.
    assert_eq!(m.cpu.reg(REG_RA), m.memory.exit_address());
    assert_eq!(m.cpu.reg(REG_SP), (before.get(REG_SP) - 16) & !0xF);
    assert_eq!(m.cpu.pc(), m.memory.exit_address() + 4);

    // vmcall starts from the initial stack pointer, whatever SP was.
    m.cpu.set_reg(REG_SP, 0x1234_5678);
    assert_eq!(m.vmcall("bump", 1).unwrap(), 11);
    assert_eq!(m.cpu.reg(REG_SP), m.memory.stack_initial() & !0xF);
}

/// Timeouts and faults in a preemption: forwarded with the registers
/// restored, or discarded by `try_preempt`. The caller's limit is kept.
#[test]
fn preempt_errors() {
    if !testutil::need_compiler() {
        return;
    }
    let mut m = machine(&build(REGS_PROGRAM));
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    m.set_max_instructions(12_345);
    let before = m.cpu.registers().clone();

    let err = m.preempt(1000, "spin", 1_000_000).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
    assert_eq!(m.cpu.registers(), &before);
    assert_eq!(m.max_instructions(), 12_345);

    let err = m.preempt(10_000, "crash", ()).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ProtectionFault);
    assert_eq!(m.cpu.registers(), &before);

    // Discarded errors: the restored a0.
    let r = m.try_preempt(1000, "spin", 1_000_000, true).unwrap();
    assert_eq!(r, before.get(REG_ARG0));
    assert_eq!(m.cpu.registers(), &before);
    let r = m.try_preempt(1000, "crash", (), true).unwrap();
    assert_eq!(r, before.get(REG_ARG0));
    // Without restoring: whatever the failed call left.
    m.try_preempt(1000, "spin", 1_000_000, false).unwrap();
    assert_ne!(m.cpu.registers(), &before);
    assert_eq!(m.max_instructions(), 12_345);

    // A preemption that finishes in time.
    assert_eq!(m.preempt(100_000, "spin", 100).unwrap(), 100);
}

/// Preemption from a system call resumes the interrupted program exactly,
/// and its instructions count against the program's own limit.
#[test]
fn preemption_shares_the_instruction_budget() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	long syscall1(long n, long arg0) {
		register long a0 __asm__("a0") = arg0;
		register long syscall_id __asm__("a7") = n;
		__asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));
		return a0;
	}
	__attribute__((used, retain))
	long spin(long n) {
		for (volatile long i = 0; i < n; i++);
		return n;
	}
	int main() {
		long sum = 0;
		for (int i = 0; i < 100; i++)
			sum += syscall1(500, i);
		return sum == 4950 ? 42 : 1;
	}"#,
    );
    let handler = |m: &mut Machine| -> libriscv::Result<()> {
        let arg = m.sysarg(0);
        m.preempt(1_000_000, "spin", 1000)?;
        // The interrupted program's registers are back.
        assert_eq!(m.sysarg(0), arg);
        Ok(())
    };
    let mut m = machine(&binary);
    m.install_syscall_handler(500, handler);
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(m.return_value(), 42);
    let total = m.instruction_counter();

    // Without the preemptions, the program takes far fewer instructions.
    let mut plain = machine(&binary);
    plain.install_syscall_handler(500, |_| Ok(()));
    plain.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(plain.return_value(), 42);
    assert!(total > plain.instruction_counter() + 100 * 1000);

    // So a limit between the two is reached.
    let mut limited = machine(&binary);
    limited.install_syscall_handler(500, handler);
    let limit = (total + plain.instruction_counter()) / 2;
    let err = limited.simulate(limit).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
}

/// `try_vmcall`: reaching the limit returns, without an error.
#[test]
fn try_vmcall_timeout() {
    if !testutil::need_compiler() {
        return;
    }
    let mut m = machine(&build(REGS_PROGRAM));
    m.try_vmcall(1000, "spin", 1_000_000).unwrap();
    assert!(m.instruction_limit_reached());
    let r = m.try_vmcall(1_000_000, "spin", 10).unwrap();
    assert_eq!(r, 10);
    assert!(!m.instruction_limit_reached());
    let err = m.vmcall_limited(1000, "spin", 1_000_000).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
    // Unknown functions are address 0, which can't be executed.
    let err = m.vmcall("no_such_function", ()).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ExecutionSpaceProtectionFault);
}

/// `default_exit_function`: VM calls return to a function in the program.
#[test]
fn default_exit_function() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
	__attribute__((used, retain, naked))
	void fast_exit() {
		__asm__ volatile (".insn i SYSTEM, 0, x0, x0, 0x7ff");
	}
	__attribute__((used, retain))
	long twice(long n) {
		return 2 * n;
	}
	int main() {
		return 0;
	}"#,
    );
    let mut m = machine_with(
        &binary,
        MachineOptions {
            default_exit_function: Some("fast_exit".into()),
            ..Default::default()
        },
    );
    let fast_exit = m.address_of("fast_exit").unwrap();
    assert_ne!(fast_exit, 0);
    assert_eq!(m.memory.exit_address(), fast_exit);
    assert_eq!(m.vmcall("twice", 21).unwrap(), 42);
    assert_eq!(m.cpu.pc(), fast_exit + 4);
    // An unknown symbol falls back to the host page.
    let m = machine_with(
        &binary,
        MachineOptions {
            default_exit_function: Some("no_such_function".into()),
            ..Default::default()
        },
    );
    assert_ne!(m.memory.exit_address(), fast_exit);
    assert_ne!(m.memory.exit_address(), 0);
}

/// Arguments keep counting up the register file past a7 (as in C++);
/// past x31 is an error, not a panic.
#[test]
fn too_many_arguments() {
    if !testutil::need_compiler() {
        return;
    }
    struct Ints(u64);
    impl libriscv::VmCallArg for Ints {
        fn pass(&self, m: &mut Machine, regs: &mut libriscv::ArgRegs) -> libriscv::Result<()> {
            for i in 0..self.0 {
                regs.int(m, i + 1)?;
            }
            Ok(())
        }
    }
    let mut m = machine(&build(REGS_PROGRAM));
    let args = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    // a0..a7, s2..s5: still inside the register file.
    m.setup_call(&args).unwrap();
    for r in 10..22u32 {
        assert_eq!(m.cpu.reg(r), u64::from(r) - 9);
    }
    // a0..x31 is 22 registers.
    m.setup_call(&Ints(22)).unwrap();
    assert_eq!(m.cpu.reg(31), 22);
    let err = m.setup_call(&Ints(23)).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::IllegalOperation);
    assert_eq!(m.cpu.reg(0), 0);
}

/// Preemption from a system call while background JIT compilation is
/// pending: the nested run has its own slices, and the outer run's
/// slicing (which must not be observable) continues afterwards.
#[cfg(feature = "jit")]
#[test]
fn vm_call_and_preemption_jit() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(PREEMPT_PROGRAM);
    for (background, slice) in [(false, 1), (true, 1), (true, 7), (true, 1000)] {
        let options = MachineOptions {
            jit: true,
            jit_background: background,
            jit_background_slice: slice,
            ..Default::default()
        };
        vm_call_and_preemption_in(machine_with(&binary, options));
    }
    // Same counts as the interpreter.
    let count = |options: MachineOptions| {
        let mut m = machine_with(&binary, options);
        m.set_printer(|_, _| {});
        m.install_syscall_handler(500, preempt_handler);
        m.simulate(MAX_INSTRUCTIONS).unwrap();
        (m.instruction_counter(), m.return_value())
    };
    let interp = count(MachineOptions::default());
    for slice in [1, 3, 64] {
        let jit = count(MachineOptions {
            jit: true,
            jit_background: true,
            jit_background_slice: slice,
            ..Default::default()
        });
        assert_eq!(jit, interp, "slice {slice}");
    }
}

/// A preemption gets its own slicing state. Here native code is installed
/// from the handler, so the nested run isn't sliced while the outer one
/// is, and the nested limit is exactly the outer one: the nested run must
/// still see (and run to) its own limit, not the end of the outer slice.
#[cfg(feature = "jit")]
#[test]
fn preemption_during_background_slice() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(PREEMPT_PROGRAM);
    let mut m = machine_with(
        &binary,
        MachineOptions {
            jit: true,
            jit_background: true,
            jit_background_slice: 1,
            ..Default::default()
        },
    );
    catch_hello(&mut m);
    m.install_syscall_handler(500, |m| {
        m.wait_for_jit();
        let remaining = m.max_instructions() - m.instruction_counter();
        let func = m.address_of("preempt").unwrap();
        m.preempt(remaining, func, "Hello World!".len())?;
        Ok(())
    });
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    assert!(hello_world(&m));
    assert_eq!(m.return_value() as i32, 666);
}
