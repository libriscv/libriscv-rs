/*
 * CoreMark port for freestanding RV64 guests under libriscv: the platform
 * hooks (timing, output, entry point) on top of the minimal system calls
 * write (64) and exit (93). See core_portme.h.
 */
#include "coremark.h"
#include <stdarg.h>

#if VALIDATION_RUN
volatile ee_s32 seed1_volatile = 0x3415;
volatile ee_s32 seed2_volatile = 0x3415;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PERFORMANCE_RUN
volatile ee_s32 seed1_volatile = 0x0;
volatile ee_s32 seed2_volatile = 0x0;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PROFILE_RUN
volatile ee_s32 seed1_volatile = 0x8;
volatile ee_s32 seed2_volatile = 0x8;
volatile ee_s32 seed3_volatile = 0x8;
#endif
volatile ee_s32 seed4_volatile = ITERATIONS;
volatile ee_s32 seed5_volatile = 0;

static long sys(long n, long a, long b, long c)
{
    register long a0 asm("a0") = a;
    register long a1 asm("a1") = b;
    register long a2 asm("a2") = c;
    register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

/* Freestanding: the compiler may still emit calls to these. */
void *memset(void *dst, int c, unsigned long n)
{
    unsigned char *d = dst;
    while (n--)
        *d++ = (unsigned char)c;
    return dst;
}
void *memcpy(void *dst, const void *src, unsigned long n)
{
    unsigned char *d = dst;
    const unsigned char *s = src;
    while (n--)
        *d++ = *s++;
    return dst;
}

/* Time: retired instructions, at a nominal 100 kHz, so that even short
   validation runs pass the "at least 10 secs" rule. */
#define EE_TICKS_PER_SEC 100000UL
static CORETIMETYPE start_time_val, stop_time_val;
static CORETIMETYPE instret(void)
{
    unsigned long v;
    asm volatile("rdinstret %0" : "=r"(v));
    return v;
}
void start_time(void) { start_time_val = instret(); }
void stop_time(void) { stop_time_val = instret(); }
CORE_TICKS get_time(void) { return stop_time_val - start_time_val; }
secs_ret time_in_secs(CORE_TICKS ticks)
{
    return ((secs_ret)ticks) / (secs_ret)EE_TICKS_PER_SEC;
}

ee_u32 default_num_contexts = 1;

void portable_init(core_portable *p, int *argc, char *argv[])
{
    (void)argc;
    (void)argv;
    p->portable_id = 1;
}
void portable_fini(core_portable *p) { p->portable_id = 0; }

/* A small ee_printf covering what CoreMark prints: %d %u %x %lu %s %c %f,
   with zero padding and a width. */
static char outbuf[256];
static int outlen;
static void flush(void)
{
    if (outlen)
        sys(64, 1, (long)outbuf, outlen);
    outlen = 0;
}
static void putch(char c)
{
    if (outlen == (int)sizeof(outbuf))
        flush();
    outbuf[outlen++] = c;
}
static void putnum(unsigned long v, unsigned base, int width, char pad, int neg)
{
    char tmp[24];
    int n = 0;
    do {
        tmp[n++] = "0123456789abcdef"[v % base];
        v /= base;
    } while (v);
    if (neg)
        width--;
    if (neg && pad == '0')
        putch('-');
    for (int i = n; i < width; i++)
        putch(pad);
    if (neg && pad != '0')
        putch('-');
    while (n)
        putch(tmp[--n]);
}
#if HAS_FLOAT
static void putdouble(double d)
{
    if (d < 0) {
        putch('-');
        d = -d;
    }
    unsigned long ip = (unsigned long)d;
    unsigned long frac = (unsigned long)((d - (double)ip) * 1000000.0 + 0.5);
    if (frac >= 1000000) {
        ip++;
        frac -= 1000000;
    }
    putnum(ip, 10, 0, ' ', 0);
    putch('.');
    putnum(frac, 10, 6, '0', 0);
}
#endif

int ee_printf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') {
            putch(*p);
            continue;
        }
        p++;
        char pad = ' ';
        int width = 0, lng = 0;
        if (*p == '0') {
            pad = '0';
            p++;
        }
        while (*p >= '0' && *p <= '9')
            width = width * 10 + (*p++ - '0');
        while (*p == 'l') {
            lng = 1;
            p++;
        }
        switch (*p) {
        case 'd': {
            long v = lng ? va_arg(ap, long) : va_arg(ap, int);
            putnum(v < 0 ? -(unsigned long)v : (unsigned long)v, 10, width, pad, v < 0);
            break;
        }
        case 'u':
            putnum(lng ? va_arg(ap, unsigned long) : va_arg(ap, unsigned), 10, width, pad, 0);
            break;
        case 'x':
            putnum(lng ? va_arg(ap, unsigned long) : va_arg(ap, unsigned), 16, width, pad, 0);
            break;
        case 's':
            for (const char *s = va_arg(ap, const char *); *s; s++)
                putch(*s);
            break;
        case 'c':
            putch((char)va_arg(ap, int));
            break;
#if HAS_FLOAT
        case 'f':
            putdouble(va_arg(ap, double));
            break;
#endif
        case '%':
            putch('%');
            break;
        default:
            putch('%');
            putch(*p);
            break;
        }
    }
    va_end(ap);
    flush();
    return 0;
}

int main(int argc, char *argv[]);
void _start(void)
{
    sys(93, main(0, 0), 0, 0);
}
