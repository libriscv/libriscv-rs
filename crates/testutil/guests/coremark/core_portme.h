/*
 * CoreMark port for freestanding RV64 guests under libriscv (C++ and Rust).
 * Based on the structure of CoreMark's barebones/core_portme.h
 * (Copyright 2018 EEMBC, Apache License 2.0).
 *
 * Time is the retired-instruction counter (RDINSTRET) at a nominal
 * 100 kHz, so the output is deterministic and identical under every
 * emulator that counts instructions exactly. The scores it prints are
 * therefore iterations per 100k instructions; measure speed from the host.
 */
#ifndef CORE_PORTME_H
#define CORE_PORTME_H

#ifndef HAS_FLOAT
#define HAS_FLOAT 1
#endif
#define HAS_TIME_H 0
#define USE_CLOCK 0
#define HAS_STDIO 0
#define HAS_PRINTF 0

#ifndef COMPILER_VERSION
#define COMPILER_VERSION "GCC" __VERSION__
#endif
#ifndef COMPILER_FLAGS
#define COMPILER_FLAGS "freestanding"
#endif
#define MEM_LOCATION "STACK"

typedef signed short ee_s16;
typedef unsigned short ee_u16;
typedef signed int ee_s32;
typedef double ee_f32;
typedef unsigned char ee_u8;
typedef unsigned int ee_u32;
typedef unsigned long ee_ptr_int;
typedef unsigned long ee_size_t;
#define NULL ((void *)0)

#define align_mem(x) (void *)(4 + (((ee_ptr_int)(x)-1) & ~3))

#define CORETIMETYPE unsigned long
typedef unsigned long CORE_TICKS;

#define SEED_METHOD SEED_VOLATILE
#define MEM_METHOD MEM_STACK
#define MULTITHREAD 1
#define USE_PTHREAD 0
#define USE_FORK 0
#define USE_SOCKET 0
#define MAIN_HAS_NOARGC 0
#define MAIN_HAS_NORETURN 0

extern ee_u32 default_num_contexts;

typedef struct CORE_PORTABLE_S
{
    ee_u8 portable_id;
} core_portable;

void portable_init(core_portable *p, int *argc, char *argv[]);
void portable_fini(core_portable *p);

#if !defined(PROFILE_RUN) && !defined(PERFORMANCE_RUN) \
    && !defined(VALIDATION_RUN)
#if (TOTAL_DATA_SIZE == 1200)
#define PROFILE_RUN 1
#elif (TOTAL_DATA_SIZE == 2000)
#define PERFORMANCE_RUN 1
#else
#define VALIDATION_RUN 1
#endif
#endif

int ee_printf(const char *fmt, ...);

#endif /* CORE_PORTME_H */
