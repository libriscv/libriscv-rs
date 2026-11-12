//! Linux guests: the system call set, the initial stack, threads, signals
//! and the file system sandbox.
//!
//! Ports of `tests/unit/security.cpp` (C1-C3, C5-C7), the Linux cases of
//! `tests/unit/protections.cpp` and `tests/unit/threads/test_threads.cpp`.
//! C4 (the arena tail guard) is covered by the memory unit tests. The C++
//! tests use 8 MiB of memory with virtual paging; without paging the main
//! stack (above 16 MiB of brk space) must be inside the arena, so these use
//! the default size.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::{REG_ARG0, REG_ARG7};
use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 10_000_000;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];
const ENOSYS: i64 = 38;
const EPERM: i64 = 1;
const EBADF: i64 = 9;
const EINVAL: i64 = 22;

fn build(code: &str) -> Vec<u8> {
    testutil::build_and_load(code, "-O2 -static").unwrap_or_else(|e| panic!("{e}"))
}

/// A machine with the Linux system calls and initial stack.
fn linux_machine(binary: &[u8], filesystem: bool) -> Machine {
    let mut machine = Machine::new(binary, MachineOptions::default()).unwrap();
    machine.setup_linux_syscalls(filesystem, false);
    machine.setup_linux(&["linux_test"], &ENV).unwrap();
    machine
}

fn run(code: &str) -> i64 {
    let mut machine = linux_machine(&build(code), false);
    machine.setup_posix_threads();
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    machine.return_value() as i64
}

/// Call system call `number` directly with up to four arguments.
fn syscall(m: &mut Machine, number: u64, args: &[u64]) -> i64 {
    for (i, &a) in args.iter().enumerate() {
        m.cpu.set_reg(REG_ARG0 + i as u32, a);
    }
    m.system_call(number).unwrap();
    m.return_value() as i64
}

const HELLO: &str = "int main() { return 0; }";

#[test]
fn ioctl_with_unknown_request_is_denied() {
    // C1: no ioctl request reaches the host.
    if !testutil::need_compiler() {
        return;
    }
    let mut machine = linux_machine(&build(HELLO), true);
    machine.fds_mut().unwrap().permit_filesystem = true;
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    let r = syscall(&mut machine, 29, &[1, 0x1234, 0xDEAD_0000]);
    assert!(r == -ENOSYS || r == -EPERM, "ioctl result: {r}");
    // TCGETS on stdout: not a terminal as far as the guest knows.
    let r = syscall(&mut machine, 29, &[1, 0x5401, 0xDEAD_0000]);
    assert_eq!(r, -ENOSYS);
    // A filter can deny.
    machine.fds_mut().unwrap().filter_ioctl = Some(Box::new(|_| false));
    assert_eq!(syscall(&mut machine, 29, &[1, 0x5401, 0]), -EPERM);
}

#[test]
fn cross_page_memory_access() {
    // C2
    if !testutil::need_compiler() {
        return;
    }
    let code = r#"
    #include <stdint.h>
    #include <string.h>
    #include <sys/mman.h>
    int main() {
        char* base = (char*)mmap(NULL, 8192, PROT_READ | PROT_WRITE,
            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (base == (void*)-1)
            return -1;
        uintptr_t boundary = ((uintptr_t)base + 4096) & ~(uintptr_t)0xFFF;
        char* cross = (char*)(boundary - 4);
        uint64_t write_val = 0xDEADBEEFCAFEBABEULL, read_val;
        memcpy(cross, &write_val, 8);
        memcpy(&read_val, cross, 8);
        if (read_val != write_val)
            return -2;
        uint16_t w16 = 0xABCD, r16;
        memcpy((char*)(boundary - 1), &w16, 2);
        memcpy(&r16, (char*)(boundary - 1), 2);
        if (r16 != w16)
            return -3;
        return 42;
    }"#;
    assert_eq!(run(code), 42);
}

#[test]
fn faccessat_respects_permission_checks() {
    // C3: the guest can't probe the host file system.
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
        long syscall(long n, ...);
        int main() {
            long ret = syscall(48, -100, "/etc/passwd", 0, 0);
            if (ret == 0)
                return 1;  // The guest could probe the host
            if (ret == -1)
                return 42; // Denied
            return (int)ret;
        }"#,
    );
    // Without file descriptors
    let mut machine = linux_machine(&binary, false);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 42);
    // With file descriptors, but the file system not permitted
    let mut machine = linux_machine(&binary, true);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 42);
    // Permitted, but no filter: still denied outside proxy mode
    let mut machine = linux_machine(&binary, true);
    machine.fds_mut().unwrap().permit_filesystem = true;
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 42);
}

#[test]
fn nanosleep_is_clamped_and_penalized() {
    // C5
    if !testutil::need_compiler() {
        return;
    }
    let mut machine = linux_machine(&build(HELLO), false);
    let ts = machine.memory.mmap_allocate(4096);
    let mut req = [0u8; 16];
    req[..8].copy_from_slice(&2i64.to_le_bytes());
    machine.copy_to_guest(ts, &req).unwrap();
    let t0 = std::time::Instant::now();
    assert_eq!(syscall(&mut machine, 101, &[ts, 0]), 0);
    let elapsed = t0.elapsed();
    // Clamped: less than the requested 2 seconds
    assert!(elapsed.as_millis() < 1900, "{elapsed:?}");
    // Penalized: 2 seconds requested = 2M instructions
    assert!(machine.instruction_counter() >= 2_000_000);
    // Negative durations are rejected without sleeping
    req[..8].copy_from_slice(&0i64.to_le_bytes());
    req[8..].copy_from_slice(&(-1i64).to_le_bytes());
    machine.copy_to_guest(ts, &req).unwrap();
    assert_eq!(syscall(&mut machine, 101, &[ts, 0]), -EINVAL);
}

#[test]
fn getcwd_and_pselect_without_file_descriptors() {
    // C6, C7
    if !testutil::need_compiler() {
        return;
    }
    let mut machine = linux_machine(&build(HELLO), false);
    let buf = machine.memory.mmap_allocate(4096);
    assert_eq!(syscall(&mut machine, 17, &[buf, 4096]), -EBADF);
    assert_eq!(syscall(&mut machine, 72, &[]), 0);
    // With file descriptors: the fake working directory, if it fits.
    let mut machine = linux_machine(&build(HELLO), true);
    assert_eq!(syscall(&mut machine, 17, &[buf, 4096]), 6);
    assert_eq!(machine.memory.memview(buf, 6).unwrap(), b"/home\0");
    assert_eq!(syscall(&mut machine, 17, &[buf, 3]), -34);
}

#[test]
fn map_fixed_noreplace_reserves_an_exact_address() {
    if !testutil::need_compiler() {
        return;
    }
    let code = r#"
    #include <errno.h>
    #include <sys/mman.h>
    #ifndef MAP_FIXED_NOREPLACE
    #define MAP_FIXED_NOREPLACE 0x100000
    #endif
    int main() {
        const unsigned long SIZE = 0x100000UL;
        char *probe = mmap(0, SIZE, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (probe == MAP_FAILED)
            return -1;
        char *ADDR = probe + 16 * SIZE;
        void *first = mmap(ADDR, SIZE, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (first != (void *)ADDR)
            return -2;
        *(volatile int *)first = 1234;
        void *again = mmap(ADDR, SIZE, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (again != MAP_FAILED || errno != EEXIST)
            return -4;
        char *anon = mmap(0, SIZE, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (anon == MAP_FAILED)
            return -5;
        if (anon + SIZE > ADDR && anon < ADDR + SIZE)
            return -6;
        if (*(volatile int *)first != 1234)
            return -7;
        return 666;
    }"#;
    assert_eq!(run(code), 666);
}

#[test]
fn mmap_munmap_reuse_and_zeroing() {
    if !testutil::need_compiler() {
        return;
    }
    let code = r#"
    #define _GNU_SOURCE
    #include <string.h>
    #include <sys/mman.h>
    int main() {
        char *a = mmap(0, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        char *b = mmap(0, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (a == MAP_FAILED || b == MAP_FAILED || b != a + 65536)
            return -1;
        memset(a, 0x55, 65536);
        /* Not the last mapping: goes to the free list, and is reused */
        munmap(a, 65536);
        char *c = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (c != a)
            return -2;
        for (int i = 0; i < 4096; i++)
            if (c[i] != 0) return -3;
        /* The last mapping can grow in place */
        char *d = mremap(b, 65536, 131072, MREMAP_MAYMOVE);
        if (d != b)
            return -4;
        if (mmap((void*)1, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) != MAP_FAILED)
            return -5;
        return 666;
    }"#;
    assert_eq!(run(code), 666);
}

#[test]
fn malloc_free_and_brk() {
    if !testutil::need_compiler() {
        return;
    }
    let code = r#"
    #include <stdlib.h>
    #include <string.h>
    int main() {
        /* Small allocations come from brk, large ones from mmap */
        for (int round = 0; round < 3; round++) {
            void *small[64], *large[8];
            for (int i = 0; i < 64; i++) { small[i] = malloc(100 + i); memset(small[i], i, 100 + i); }
            for (int i = 0; i < 8; i++) { large[i] = malloc(1 << 20); memset(large[i], i, 1 << 20); }
            for (int i = 0; i < 64; i++) free(small[i]);
            for (int i = 0; i < 8; i++) free(large[i]);
        }
        return 666;
    }"#;
    assert_eq!(run(code), 666);
}

#[test]
fn file_system_sandbox() {
    if !testutil::need_compiler() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("libriscv-linux-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("data.txt");
    std::fs::write(&file, b"sandboxed contents").unwrap();
    let path = file.to_str().unwrap().to_owned();
    let code = format!(
        r#"
        #include <fcntl.h>
        #include <stdio.h>
        #include <string.h>
        #include <unistd.h>
        #include <sys/stat.h>
        int main() {{
            int fd = open("{path}", O_RDONLY);
            if (fd < 0) return 1;
            char buf[64] = {{0}};
            if (read(fd, buf, sizeof(buf)) != 18) return 2;
            if (strcmp(buf, "sandboxed contents") != 0) return 3;
            if (lseek(fd, 10, SEEK_SET) != 10) return 4;
            if (pread(fd, buf, 4, 0) != 4 || memcmp(buf, "sand", 4) != 0) return 5;
            struct stat st;
            if (fstat(fd, &st) != 0 || st.st_size != 18) return 6;
            int fd2 = dup(fd);
            if (fd2 < 0 || fd2 == fd) return 7;
            if (close(fd) != 0 || close(fd) == 0) return 8;
            if (read(fd2, buf, 4) != 4 || memcmp(buf, "cont", 4) != 0) return 9;
            /* Writing is refused outside proxy mode */
            if (open("{path}", O_WRONLY | O_TRUNC) >= 0) return 10;
            if (write(fd2, "x", 1) >= 0) return 11;
            /* Other files are filtered */
            if (open("/etc/passwd", O_RDONLY) >= 0) return 12;
            return 666;
        }}"#
    );
    let binary = build(&code);
    let allowed = path.clone();
    let mut machine = linux_machine(&binary, true);
    let fds = machine.fds_mut().unwrap();
    fds.permit_filesystem = true;
    fds.filter_open = Some(Box::new(move |p: &mut String| *p == allowed));
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    // The file is untouched.
    assert_eq!(std::fs::read(&file).unwrap(), b"sandboxed contents");

    // Without permit_filesystem, nothing opens.
    let mut machine = linux_machine(&binary, true);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn signals_and_sigreturn() {
    if !testutil::need_compiler() {
        return;
    }
    let code = r#"
    #include <signal.h>
    #include <string.h>
    static volatile int caught = 0;
    static char altstack[16384];
    static void handler(int sig) { caught += sig; }
    int main() {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = handler;
        if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;
        raise(SIGUSR1);
        if (caught != SIGUSR1) return 2;
        /* Now on an alternate stack */
        stack_t ss = { .ss_sp = altstack, .ss_size = sizeof(altstack), .ss_flags = 0 };
        if (sigaltstack(&ss, 0) != 0) return 3;
        sa.sa_flags = SA_ONSTACK;
        if (sigaction(SIGUSR2, &sa, 0) != 0) return 4;
        raise(SIGUSR2);
        if (caught != SIGUSR1 + SIGUSR2) return 5;
        return 666;
    }"#;
    assert_eq!(run(code), 666);
}

#[test]
fn posix_threads() {
    // tests/unit/threads/test_threads.cpp
    if !testutil::need_compiler() {
        return;
    }
    let source = testutil::cpp_source_dir().join("tests/unit/threads/test_threads.cpp");
    let Ok(code) = std::fs::read_to_string(&source) else {
        eprintln!("SKIPPED: {} not found", source.display());
        return;
    };
    let binary = testutil::build_and_load_cxx(&code, "-O2 -static -pthread")
        .unwrap_or_else(|e| panic!("{e}"));
    let mut machine = linux_machine(&binary, false);
    machine.setup_posix_threads();
    static OUTPUT: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());
    machine.set_printer(|_, data| OUTPUT.lock().unwrap().extend_from_slice(data));
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    let output = String::from_utf8(OUTPUT.lock().unwrap().clone()).unwrap();
    assert_eq!(machine.return_value() as i32, 123_666_123, "{output}");
    assert!(output.contains("Thread depth 20 / 20"), "{output}");
    assert!(output.contains("C++ thread arguments are OK"), "{output}");
    // Everything but the main thread is gone.
    assert_eq!(machine.threads().unwrap().len(), 1);
}

#[test]
fn thread_exit_and_futex_wake() {
    if !testutil::need_compiler() {
        return;
    }
    let code = r#"
    #include <pthread.h>
    #include <stdatomic.h>
    static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    static int ready = 0, sum = 0;
    static void* worker(void* arg) {
        pthread_mutex_lock(&mtx);
        while (!ready) pthread_cond_wait(&cond, &mtx);
        sum += (int)(long)arg;
        pthread_mutex_unlock(&mtx);
        return arg;
    }
    int main() {
        pthread_t t[8];
        for (long i = 0; i < 8; i++)
            if (pthread_create(&t[i], 0, worker, (void*)(i + 1)) != 0) return 1;
        pthread_mutex_lock(&mtx);
        ready = 1;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mtx);
        long total = 0;
        for (int i = 0; i < 8; i++) {
            void* ret;
            pthread_join(t[i], &ret);
            total += (long)ret;
        }
        return (sum == 36 && total == 36) ? 666 : 2;
    }"#;
    assert_eq!(run(code), 666);
}

#[test]
fn too_many_threads_is_an_error() {
    if !testutil::need_compiler() {
        return;
    }
    let code = r#"
    #include <pthread.h>
    #include <sched.h>
    static void* spin(void* arg) { for (;;) sched_yield(); return arg; }
    int main() {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 65536);
        for (int i = 0; i < 100; i++) {
            pthread_t t;
            if (pthread_create(&t, &attr, spin, 0) != 0) return 1;
        }
        return 2;
    }"#;
    let mut machine = linux_machine(&build(code), false);
    machine.setup_posix_threads();
    let result = machine.simulate(MAX_INSTRUCTIONS);
    let err = result.map(|()| machine.return_value() as i64).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::InvalidProgram, "{err}");
    assert_eq!(err.message, "Too many threads");
}

#[test]
fn initial_stack_layout() {
    if !testutil::need_compiler() {
        return;
    }
    let machine = linux_machine(&build(HELLO), false);
    let sp = machine.cpu.reg(libriscv::registers::REG_SP);
    assert_eq!(sp % 16, 0);
    assert_eq!(machine.memory.stack_initial(), sp);
    let word = |m: &Machine, addr: u64| m.memory.read_u64(addr).unwrap();
    // argc, argv[0], NULL, envp..., NULL, auxv
    assert_eq!(word(&machine, sp), 1);
    let argv0 = word(&machine, sp + 8);
    assert_eq!(machine.memory.memstring(argv0).unwrap(), "linux_test");
    assert_eq!(word(&machine, sp + 16), 0);
    for (i, env) in ENV.iter().enumerate() {
        let p = word(&machine, sp + 24 + 8 * i as u64);
        assert_eq!(machine.memory.memstring(p).unwrap(), *env);
    }
    let mut auxv = sp + 24 + 8 * ENV.len() as u64 + 8;
    let mut seen = std::collections::HashMap::new();
    loop {
        let (key, value) = (word(&machine, auxv), word(&machine, auxv + 8));
        if key == libriscv::auxv::AT_NULL {
            break;
        }
        seen.insert(key, value);
        auxv += 16;
    }
    assert_eq!(seen[&libriscv::auxv::AT_PAGESZ], 4096);
    assert_eq!(
        seen[&libriscv::auxv::AT_ENTRY],
        machine.memory.start_address()
    );
    assert_eq!(seen[&libriscv::auxv::AT_PHENT], 56);
    let platform = seen[&libriscv::auxv::AT_PLATFORM];
    assert_eq!(machine.memory.memstring(platform).unwrap(), "RISC-V 64-bit");
    // The program headers are where AT_PHDR says.
    let phdr = seen[&libriscv::auxv::AT_PHDR];
    let phnum = seen[&libriscv::auxv::AT_PHNUM];
    let binary = machine.memory.binary().to_vec();
    let phoff = u64::from_le_bytes(binary[32..40].try_into().unwrap()) as usize;
    let len = phnum as usize * 56;
    assert_eq!(
        machine.memory.memview(phdr, len as u64).unwrap(),
        &binary[phoff..phoff + len]
    );
}

/// Every system call with random arguments, on machines with and without
/// file descriptors and threads: nothing may panic, and failures are
/// `MachineError`s. (The file system is permitted, but every path is
/// filtered out.)
#[test]
fn system_calls_with_random_arguments() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(HELLO);
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for variant in 0..4 {
        let mut machine = linux_machine(&binary, variant & 1 != 0);
        if variant & 2 != 0 {
            machine.setup_posix_threads();
        }
        if let Some(fds) = machine.fds_mut() {
            fds.permit_filesystem = true;
            fds.filter_open = Some(Box::new(|_| false));
            fds.filter_readlink = Some(Box::new(|_| false));
        }
        let heap = machine.memory.mmap_allocate(1 << 16);
        machine.set_printer(|_, _| {});
        for number in 0..libriscv::SYSCALLS_MAX as u64 {
            // Sleeping, and stopping the machine, are fine but slow.
            if matches!(number, 101 | 115) {
                continue;
            }
            for _ in 0..20 {
                for reg in REG_ARG0..=REG_ARG0 + 6 {
                    let v = match next() % 4 {
                        0 => next(),
                        1 => heap + next() % (1 << 16),
                        2 => next() % 4096,
                        _ => (next() % 64).wrapping_sub(32),
                    };
                    machine.cpu.set_reg(reg, v);
                }
                machine.cpu.set_reg(REG_ARG7, number);
                let _ = machine.system_call(number);
            }
        }
    }
}
