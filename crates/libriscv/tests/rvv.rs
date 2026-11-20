//! Port of `tests/unit/rvv.cpp`: RVV 1.0 through hand-written vector
//! assembly, which any toolchain whose assembler knows the ratified vector
//! extension can build (`-march=rv64gcv`).
//!
//! The guest is freestanding, so instead of `setup_linux` it runs with the
//! minimal system calls; it exits with the address of its results. Besides
//! C++'s two simulators, the same guest also runs under the JIT, which
//! leaves every vector instruction to the interpreter.

#![cfg(feature = "ext-v")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 16_000_000;
/// Result slots in the guest's `out[]` array.
const RESULTS: u64 = 224;

// -mno-relax: no crt0 sets up gp in a freestanding guest, so GP-relative
// addressing must not be used.
const ARGS: &str = "-std=gnu11 -O1 -static -march=rv64gcv -mabi=lp64d -nostdlib \
                    -ffreestanding -mno-relax -Wl,-z,separate-code";

const GUEST: &str = r#"
typedef unsigned long ulong;

/* Result words, checked by the host. */
volatile ulong out[224];
static ulong idx = 0;
static void put(ulong v) { out[idx++] = v; }

static const unsigned int isrc1[8] = {1,2,3,4,5,6,7,8};
static const unsigned int isrc2[8] = {10,200,30,4000,50,600000,70,4294967295u};
static unsigned int idst[16];

static ulong fbits(float f) { union { float f; ulong u; } c; c.u = 0; c.f = f; return c.u; }

/* 1. vsetvli clamping with VLEN=256 */
static void t_vsetvli(void)
{
	ulong vl, avl, vt;
	avl = 100; asm volatile("vsetvli %0, %1, e32, m1" : "=r"(vl) : "r"(avl)); put(vl);
	avl = 3;   asm volatile("vsetvli %0, %1, e32, m1" : "=r"(vl) : "r"(avl)); put(vl);
	avl = 100; asm volatile("vsetvli %0, %1, e8, m1"  : "=r"(vl) : "r"(avl)); put(vl);
	avl = 100; asm volatile("vsetvli %0, %1, e64, m2" : "=r"(vl) : "r"(avl)); put(vl);
	avl = 100; asm volatile("vsetvli %0, %1, e32, mf2": "=r"(vl) : "r"(avl)); put(vl);
	avl = 100; asm volatile("vsetvli %0, %1, e16, m4" : "=r"(vl) : "r"(avl)); put(vl);
	avl = 100; asm volatile("vsetvli %0, %1, e64, mf8": "=r"(vl) : "r"(avl)); put(vl);
	/* rs1=x0, rd!=x0: VLMAX */
	asm volatile("vsetvli %0, x0, e16, m1" : "=r"(vl)); put(vl);
	/* vsetivli */
	asm volatile("vsetivli %0, 5, e32, m1" : "=r"(vl)); put(vl);
	asm volatile("vsetivli %0, 31, e32, m1" : "=r"(vl)); put(vl);
	/* vsetvl */
	avl = 100; vt = 0; /* e8, m1 */
	asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vt)); put(vl);
	/* vsetvl with reserved vlmul=100: vill, rd = 0 */
	vt = 4;
	asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vt)); put(vl);
}

/* 2. Integer vector arithmetic */
static void t_int_ops(void)
{
	ulong v;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(isrc1));
	asm volatile("vle32.v v2, (%0)" :: "r"(isrc2));
	asm volatile("vadd.vv v3, v1, v2");
	asm volatile("vsub.vv v4, v2, v1");
	asm volatile("vand.vv v5, v1, v2");
	asm volatile("vor.vv  v6, v1, v2");
	asm volatile("vxor.vv v7, v1, v2");
	asm volatile("vminu.vv v8, v1, v2");
	asm volatile("vsll.vi v9, v1, 4");
	asm volatile("vsrl.vi v10, v2, 3");
	asm volatile("vrsub.vi v11, v1, 15");
	asm volatile("vadd.vi  v12, v1, -7");
	asm volatile("vmv.x.s %0, v3" : "=r"(v)); put(v);        /* 1+10 */
	asm volatile("vse32.v v4, (%0)" :: "r"(&idst[0]) : "memory");
	put(idst[0]); put(idst[1]); put(idst[7]);                 /* vsub */
	asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v);         /* vand */
	asm volatile("vmv.x.s %0, v6" : "=r"(v)); put(v);         /* vor */
	asm volatile("vrgather.vi v13, v7, 1");
	asm volatile("vmv.x.s %0, v13" : "=r"(v)); put(v);        /* vxor */
	asm volatile("vmv.x.s %0, v8" : "=r"(v)); put(v);         /* vminu */
	asm volatile("vmv.x.s %0, v9" : "=r"(v)); put(v);         /* vsll */
	asm volatile("vmv.x.s %0, v10" : "=r"(v)); put(v);        /* vsrl */
	asm volatile("vmv.x.s %0, v11" : "=r"(v)); put(v);        /* vrsub */
	asm volatile("vmv.x.s %0, v12" : "=r"(v)); put(v);        /* vadd.vi */
	asm volatile("vrgather.vi v13, v3, 6");
	asm volatile("vmv.x.s %0, v13" : "=r"(v)); put(v);        /* vadd[6] */
}

/* 3. Masked operations, tail-undisturbed */
static void t_masked(void)
{
	ulong v;
	ulong beef = 0xBEEF, pat = 0x1234, five = 5;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(isrc1));
	asm volatile("vle32.v v2, (%0)" :: "r"(isrc2));
	/* v0 = (isrc1 < 5): bits 0-3 set */
	asm volatile("vmsltu.vx v0, v1, %0" :: "r"(five));
	/* masked vadd into pre-filled v20 */
	asm volatile("vmv.v.x v20, %0" :: "r"(beef));
	asm volatile("vadd.vv v20, v1, v2, v0.t");
	asm volatile("vrgather.vi v21, v20, 0");
	asm volatile("vmv.x.s %0, v21" : "=r"(v)); put(v); /* 1+10 */
	asm volatile("vrgather.vi v21, v20, 4");
	asm volatile("vmv.x.s %0, v21" : "=r"(v)); put(v); /* undisturbed 0xBEEF */
	/* vmerge.vvm: vd = mask ? vs1 : vs2 */
	asm volatile("vmerge.vvm v22, v2, v1, v0");
	asm volatile("vrgather.vi v21, v22, 0");
	asm volatile("vmv.x.s %0, v21" : "=r"(v)); put(v); /* mask=1 -> vs1 */
	asm volatile("vrgather.vi v21, v22, 4");
	asm volatile("vmv.x.s %0, v21" : "=r"(v)); put(v); /* mask=0 -> vs2 */
	/* vl=4, tail-undisturbed: elements 4-7 keep 0x1234 */
	asm volatile("vmv.v.x v21, %0" :: "r"(pat));
	asm volatile("vsetivli zero, 4, e32, m1, tu, mu");
	asm volatile("vadd.vv v21, v1, v2");
	asm volatile("vrgather.vi v23, v21, 3");
	asm volatile("vmv.x.s %0, v23" : "=r"(v)); put(v); /* 4+4000 */
	asm volatile("vrgather.vi v23, v21, 4");
	asm volatile("vmv.x.s %0, v23" : "=r"(v)); put(v); /* tail 0x1234 */
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
}

/* 4. Element widths */
static void t_widths(void)
{
	static const unsigned char bsrc[32] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
		17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
	static const unsigned short hsrc[16] = {100,200,300,400,500,600,700,800,
		900,1000,1100,1200,1300,1400,1500,1600};
	static const unsigned long long dsrc[4] = {0x100000000ull, 2, 0xdeadbeefcafeull, 42};
	ulong v;

	asm volatile("vsetivli zero, 31, e8, m1, ta, ma");
	asm volatile("vle8.v v1, (%0)" :: "r"(bsrc));
	asm volatile("vadd.vi v1, v1, 1");
	asm volatile("vrgather.vi v2, v1, 0"); asm volatile("vmv.x.s %0, v2" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v2, v1, 30"); asm volatile("vmv.x.s %0, v2" : "=r"(v)); put(v);

	asm volatile("vsetivli zero, 16, e16, m1, ta, ma");
	asm volatile("vle16.v v2, (%0)" :: "r"(hsrc));
	asm volatile("vmul.vv v3, v2, v2");
	asm volatile("vrgather.vi v4, v3, 0"); asm volatile("vmv.x.s %0, v4" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v4, v3, 15"); asm volatile("vmv.x.s %0, v4" : "=r"(v)); put(v);

	asm volatile("vsetivli zero, 4, e64, m1, ta, ma");
	asm volatile("vle64.v v4, (%0)" :: "r"(dsrc));
	asm volatile("vadd.vv v5, v4, v4");
	asm volatile("vrgather.vi v6, v5, 0"); asm volatile("vmv.x.s %0, v6" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v6, v5, 2"); asm volatile("vmv.x.s %0, v6" : "=r"(v)); put(v);
}

/* 5. Reductions, slides, gathers */
static void t_reduce_slide(void)
{
	ulong v, one = 1, s77 = 77;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(isrc1));
	asm volatile("vle32.v v4, (%0)" :: "r"(isrc2));
	/* vredsum: sum(1, 1..8) = 37 */
	asm volatile("vmv.s.x v2, %0" :: "r"(one));
	asm volatile("vredsum.vs v3, v1, v2");
	asm volatile("vmv.x.s %0, v3" : "=r"(v)); put(v);
	/* vredmaxu: max(0, isrc2) = 4294967295 */
	asm volatile("vmv.s.x v2, zero");
	asm volatile("vredmaxu.vs v5, v4, v2");
	asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v);
	/* vslidedown.vi 3 */
	asm volatile("vslidedown.vi v6, v1, 3");
	asm volatile("vrgather.vi v7, v6, 0"); asm volatile("vmv.x.s %0, v7" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v7, v6, 4"); asm volatile("vmv.x.s %0, v7" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v7, v6, 5"); asm volatile("vmv.x.s %0, v7" : "=r"(v)); put(v);
	/* vslide1down.vx: scalar 77 lands in element 7 */
	asm volatile("vslide1down.vx v8, v1, %0" :: "r"(s77));
	asm volatile("vrgather.vi v9, v8, 0"); asm volatile("vmv.x.s %0, v9" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v9, v8, 7"); asm volatile("vmv.x.s %0, v9" : "=r"(v)); put(v);
	/* vslideup.vi 2 into v10 pre-filled with 100s: elements 0,1 keep 100 */
	ulong hundo = 100;
	asm volatile("vmv.v.x v10, %0" :: "r"(hundo));
	asm volatile("vslideup.vi v10, v1, 2");
	asm volatile("vrgather.vi v9, v10, 0"); asm volatile("vmv.x.s %0, v9" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v9, v10, 1"); asm volatile("vmv.x.s %0, v9" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v9, v10, 2"); asm volatile("vmv.x.s %0, v9" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v9, v10, 3"); asm volatile("vmv.x.s %0, v9" : "=r"(v)); put(v);
	/* vrgather.vi 5, vrgather.vx 3 */
	asm volatile("vrgather.vi v11, v1, 5");
	asm volatile("vmv.x.s %0, v11" : "=r"(v)); put(v);
	ulong three = 3;
	asm volatile("vrgather.vx v12, v1, %0" :: "r"(three));
	asm volatile("vmv.x.s %0, v12" : "=r"(v)); put(v);
}

/* 6. Compares and mask operations */
static void t_compares(void)
{
	ulong v;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(isrc1));
	asm volatile("vle32.v v2, (%0)" :: "r"(isrc2));
	asm volatile("vmseq.vi v3, v1, 5");
	asm volatile("vmv.x.s %0, v3" : "=r"(v)); put(v); /* bit 4 */
	asm volatile("vmsne.vv v4, v1, v2");
	asm volatile("vmand.mm v5, v4, v4");
	asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v); /* all ones */
	asm volatile("vcpop.m %0, v3" : "=r"(v)); put(v);
	asm volatile("vfirst.m %0, v3" : "=r"(v)); put(v);
}

/* 7. LMUL=2 register groups */
static void t_lmul2(void)
{
	static const unsigned int a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	static const unsigned int b[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	ulong v;
	asm volatile("vsetivli zero, 16, e32, m2, ta, ma");
	asm volatile("vle32.v v2, (%0)" :: "r"(a));
	asm volatile("vle32.v v4, (%0)" :: "r"(b));
	asm volatile("vadd.vv v6, v2, v4");
	asm volatile("vrgather.vi v8, v6, 8");
	asm volatile("vmv.x.s %0, v8" : "=r"(v)); put(v); /* 9+8 */
	asm volatile("vrgather.vi v8, v6, 15");
	asm volatile("vmv.x.s %0, v8" : "=r"(v)); put(v); /* 16+15 */
}

/* 8. Floating-point */
static const float fsrc1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
static const float fsrc2[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
static void t_float(void)
{
	ulong v;
	float res;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(fsrc1));
	asm volatile("vle32.v v2, (%0)" :: "r"(fsrc2));
	asm volatile("vfadd.vv v3, v1, v2");
	asm volatile("vfsub.vv v4, v1, v2");
	asm volatile("vfmul.vv v5, v1, v2");
	asm volatile("vrgather.vi v6, v3, 0"); asm volatile("vfmv.f.s %0, v6" : "=f"(res)); put(fbits(res));
	asm volatile("vrgather.vi v6, v4, 0"); asm volatile("vfmv.f.s %0, v6" : "=f"(res)); put(fbits(res));
	asm volatile("vrgather.vi v6, v5, 7"); asm volatile("vfmv.f.s %0, v6" : "=f"(res)); put(fbits(res));
	/* vfredusum with seed 0: 1+..+8 = 36 */
	asm volatile("vmv.s.x v8, zero");
	asm volatile("vfredusum.vs v9, v1, v8");
	asm volatile("vfmv.f.s %0, v9" : "=f"(res)); put((ulong)res);
	/* vfmacc: v10 = v1*v2 + v10(=1.0f) = 1.5 .. 5.0 */
	ulong one_f = fbits(1.0f);
	asm volatile("vmv.v.x v10, %0" :: "r"(one_f));
	asm volatile("vfmacc.vv v10, v1, v2");
	asm volatile("vrgather.vi v6, v10, 0"); asm volatile("vfmv.f.s %0, v6" : "=f"(res)); put(fbits(res));
	asm volatile("vrgather.vi v6, v10, 7"); asm volatile("vfmv.f.s %0, v6" : "=f"(res)); put(fbits(res));
	/* vfnmacc/vfnmadd: -(a*b) - c is +0 when the product is -0 and c is +0 */
	ulong pz = 0, nz = 0x80000000ul;
	asm volatile("vmv.v.x v14, %0" :: "r"(pz));
	asm volatile("vmv.v.x v15, %0" :: "r"(nz));
	asm volatile("vmv.v.x v16, %0" :: "r"(pz));
	asm volatile("vfnmacc.vv v16, v14, v15");
	asm volatile("vmv.x.s %0, v16" : "=r"(v)); put(v & 0xffffffff);
	asm volatile("vmv.v.x v17, %0" :: "r"(nz));
	asm volatile("vfnmadd.vv v17, v14, v14");
	asm volatile("vmv.x.s %0, v17" : "=r"(v)); put(v & 0xffffffff);
	/* vfadd.vf with 10.0f */
	float ten = 10.0f;
	asm volatile("vfadd.vf v11, v1, %0" :: "f"(ten));
	asm volatile("vrgather.vi v6, v11, 0"); asm volatile("vfmv.f.s %0, v6" : "=f"(res)); put(fbits(res));
	/* vfcvt.rtz.x.f.v: 1.5 -> 1, 8.5 -> 8 */
	asm volatile("vfcvt.rtz.x.f.v v12, v3");
	asm volatile("vrgather.vi v6, v12, 0"); asm volatile("vmv.x.s %0, v6" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v6, v12, 7"); asm volatile("vmv.x.s %0, v6" : "=r"(v)); put(v);
}

/* 9. vmv and saturation */
static void t_misc(void)
{
	ulong v, eight = 8, one = 1, forty2 = 42;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vmv.v.x v1, %0" :: "r"(forty2));
	asm volatile("vadd.vx v2, v1, %0" :: "r"(eight));
	asm volatile("vmv.x.s %0, v2" : "=r"(v)); put(v); /* 50 */
	asm volatile("vle32.v v3, (%0)" :: "r"(isrc2));
	asm volatile("vsaddu.vi v4, v3, 1");
	asm volatile("vrgather.vi v5, v4, 7");
	asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v); /* saturated */
	asm volatile("vslide1up.vx v6, v3, %0" :: "r"(one));
	asm volatile("vrgather.vi v5, v6, 0"); asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v);
	asm volatile("vrgather.vi v5, v6, 1"); asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v);
	/* vmv<nr>r.v copies whole registers, ignoring vl and vtype. GCC emits
	   it whenever a vector value outlives its defining instruction. */
	asm volatile("vle32.v v8, (%0)" :: "r"(isrc1));
	asm volatile("vsetivli zero, 1, e32, m1, ta, ma"); /* vl=1 must not matter */
	asm volatile("vmv1r.v v9, v8");
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vrgather.vi v5, v9, 7"); asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v);
	/* vmv2r.v moves an aligned pair, here filled one register at a time */
	asm volatile("vle32.v v10, (%0)" :: "r"(isrc1));
	asm volatile("vle32.v v11, (%0)" :: "r"(isrc2));
	asm volatile("vmv2r.v v12, v10");
	asm volatile("vrgather.vi v5, v13, 5"); asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v);
}

/* 10. Unit-stride vle32/vse32 and element-wise float arithmetic outside the
   whole-lane configuration. The binary translator inlines those two as long as
   vtype says e32/m1 with vl == VLMAX and the instruction is unmasked; every
   case below deliberately breaks one of those conditions, or moves a lane
   through an address with no natural alignment. */
static unsigned char raw[64];
static void t_vlsu_edges(void)
{
	ulong v, pat = 0xABCD, five = 5;
	float res;

	/* Whole lane through an address that is not 4-byte aligned. */
	unsigned char *ua = &raw[1];
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(isrc1));
	asm volatile("vse32.v v1, (%0)" :: "r"(ua) : "memory");
	asm volatile("vmv.v.x v2, zero");
	asm volatile("vle32.v v2, (%0)" :: "r"(ua));
	asm volatile("vrgather.vi v3, v2, 0"); asm volatile("vmv.x.s %0, v3" : "=r"(v)); put(v); /* 1 */
	asm volatile("vrgather.vi v3, v2, 7"); asm volatile("vmv.x.s %0, v3" : "=r"(v)); put(v); /* 8 */

	/* vl < VLMAX: the load stops after 4 elements, the tail is undisturbed. */
	asm volatile("vmv.v.x v4, %0" :: "r"(pat));
	asm volatile("vsetivli zero, 4, e32, m1, tu, mu");
	asm volatile("vle32.v v4, (%0)" :: "r"(isrc2));
	asm volatile("vsetivli zero, 8, e32, m1, tu, mu");
	asm volatile("vrgather.vi v5, v4, 3"); asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v); /* 4000 */
	asm volatile("vrgather.vi v5, v4, 4"); asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v); /* 0xABCD */

	/* vl < VLMAX on a store: memory past the 4th element is left alone. */
	idst[4] = 0x5A5A;
	asm volatile("vsetivli zero, 4, e32, m1, tu, mu");
	asm volatile("vse32.v v1, (%0)" :: "r"(&idst[0]) : "memory");
	asm volatile("vsetivli zero, 8, e32, m1, tu, mu");
	put(idst[3]); put(idst[4]);                                              /* 4, 0x5A5A */

	/* Masked load: inactive elements keep their previous value. */
	asm volatile("vle32.v v6, (%0)" :: "r"(isrc1));
	asm volatile("vmsltu.vx v0, v6, %0" :: "r"(five));  /* bits 0-3 set */
	asm volatile("vmv.v.x v7, %0" :: "r"(pat));
	asm volatile("vle32.v v7, (%0), v0.t" :: "r"(isrc2));
	asm volatile("vrgather.vi v8, v7, 0"); asm volatile("vmv.x.s %0, v8" : "=r"(v)); put(v); /* 10 */
	asm volatile("vrgather.vi v8, v7, 4"); asm volatile("vmv.x.s %0, v8" : "=r"(v)); put(v); /* 0xABCD */

	/* Masked store: only the first 4 words reach memory. */
	idst[8] = 0x77; idst[12] = 0x99;
	asm volatile("vse32.v v6, (%0), v0.t" :: "r"(&idst[8]) : "memory");
	put(idst[8]); put(idst[12]);                                             /* 1, 0x99 */

	/* Float arithmetic with vl < VLMAX, tail undisturbed. */
	ulong nine_f = fbits(9.0f);
	asm volatile("vmv.v.x v9, %0" :: "r"(nine_f));
	asm volatile("vle32.v v10, (%0)" :: "r"(fsrc1));
	asm volatile("vle32.v v11, (%0)" :: "r"(fsrc2));
	asm volatile("vsetivli zero, 4, e32, m1, tu, mu");
	asm volatile("vfadd.vv v9, v10, v11");
	asm volatile("vsetivli zero, 8, e32, m1, tu, mu");
	asm volatile("vrgather.vi v12, v9, 3"); asm volatile("vfmv.f.s %0, v12" : "=f"(res)); put(fbits(res)); /* 4.5 */
	asm volatile("vrgather.vi v12, v9, 4"); asm volatile("vfmv.f.s %0, v12" : "=f"(res)); put(fbits(res)); /* 9.0 */

	/* Masked vfmul.vf: inactive elements keep their old value. */
	float two = 2.0f;
	asm volatile("vmv.v.x v13, %0" :: "r"(nine_f));
	asm volatile("vfmul.vf v13, v10, %0, v0.t" :: "f"(two));
	asm volatile("vrgather.vi v12, v13, 0"); asm volatile("vfmv.f.s %0, v12" : "=f"(res)); put(fbits(res)); /* 2.0 */
	asm volatile("vrgather.vi v12, v13, 4"); asm volatile("vfmv.f.s %0, v12" : "=f"(res)); put(fbits(res)); /* 9.0 */

	/* Same float ops at e64: the fast path is e32-only. */
	static double dsrc[4] = {1.0, 2.0, 3.0, 4.0};
	asm volatile("vsetivli zero, 4, e64, m1, ta, ma");
	asm volatile("vle64.v v14, (%0)" :: "r"(dsrc));
	asm volatile("vfadd.vv v15, v14, v14");
	asm volatile("vfmul.vv v16, v14, v14");
	asm volatile("vse64.v v15, (%0)" :: "r"(dsrc) : "memory");
	put((ulong)dsrc[3]);                                                     /* 8 */
	asm volatile("vse64.v v16, (%0)" :: "r"(dsrc) : "memory");
	put((ulong)dsrc[3]);                                                     /* 16 */
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
}

/* 11. The vector CSRs. vl, vtype and vlenb are read-only -- only vsetvl
   writes them -- while vxrm, vxsat, vstart and vcsr are writable. */
static void t_csrs(void)
{
	ulong v, bad;
	asm volatile("csrwi vxsat, 0");
	asm volatile("csrwi vxrm, 0");
	asm volatile("vsetivli zero, 7, e32, m1, ta, ma");
	asm volatile("csrr %0, vl" : "=r"(v)); put(v);
	asm volatile("csrr %0, vtype" : "=r"(v)); put(v);
	asm volatile("csrr %0, vlenb" : "=r"(v)); put(v);
	asm volatile("csrwi vxrm, 3");
	asm volatile("csrr %0, vxrm" : "=r"(v)); put(v);
	asm volatile("csrr %0, vcsr" : "=r"(v)); put(v);
	asm volatile("csrwi vxrm, 0");
	asm volatile("csrr %0, vstart" : "=r"(v)); put(v);
	/* A reserved vtype reads back as vill alone, and zeroes vl. */
	bad = 4; /* vlmul = 100 */
	asm volatile("vsetvl zero, zero, %0" :: "r"(bad));
	asm volatile("csrr %0, vtype" : "=r"(v)); put(v >> 63);
	asm volatile("csrr %0, vl" : "=r"(v)); put(v);
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
}

/* 12. The multiply family. Its eight code points sit next to each other
   and differ only in which operand is signed and which term is negated,
   so each one is checked against an answer the others cannot produce. */
static unsigned int mout[8];
static void t_multiply(void)
{
	static const int mA[8] = {2, -3, 0x40000000, 6, 0, 0, 0, 0};
	static const int mB[8] = {5, -7, 4, 7, 0, 0, 0, 0};
	ulong hundred = 100;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(mA));
	asm volatile("vle32.v v2, (%0)" :: "r"(mB));

	asm volatile("vmul.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]); put(mout[2]); put(mout[3]);
	asm volatile("vmulh.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[1]); put(mout[2]);
	asm volatile("vmulhu.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[1]);
	asm volatile("vmulhsu.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[1]);

	/* The four accumulating forms, all with vd = 100, vs1 = 5, vs2 = 2. */
	asm volatile("vmv.v.x v4, %0" :: "r"(hundred));
	asm volatile("vmacc.vv v4, v2, v1");
	asm volatile("vse32.v v4, (%0)" :: "r"(mout) : "memory"); put(mout[0]);
	asm volatile("vmv.v.x v4, %0" :: "r"(hundred));
	asm volatile("vnmsac.vv v4, v2, v1");
	asm volatile("vse32.v v4, (%0)" :: "r"(mout) : "memory"); put(mout[0]);
	asm volatile("vmv.v.x v4, %0" :: "r"(hundred));
	asm volatile("vmadd.vv v4, v2, v1");
	asm volatile("vse32.v v4, (%0)" :: "r"(mout) : "memory"); put(mout[0]);
	asm volatile("vmv.v.x v4, %0" :: "r"(hundred));
	asm volatile("vnmsub.vv v4, v2, v1");
	asm volatile("vse32.v v4, (%0)" :: "r"(mout) : "memory"); put(mout[0]);
}

/* 13. Integer divide, including the answers the ISA defines for division
   by zero and for the one signed overflow. */
static void t_divide(void)
{
	static const int dA[8] = {100, -100, 7, (int)0x80000000, 5, 0, 13, 1};
	static const int dB[8] = {7, 7, 0, -1, 0, 0, 4, 1};
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(dA));
	asm volatile("vle32.v v2, (%0)" :: "r"(dB));
	asm volatile("vdivu.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]); put(mout[2]); put(mout[3]);
	asm volatile("vdiv.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]); put(mout[2]); put(mout[3]);
	asm volatile("vremu.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[1]); put(mout[2]);
	asm volatile("vrem.vv v3, v1, v2");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[1]); put(mout[2]); put(mout[3]);
}

/* 14. Index generation, prefix count, compress, and the three mask scans
   -- which differ from one another only around the first set bit. */
static void t_permute2(void)
{
	ulong v, two = 2;
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(isrc1));    /* 1..8 */
	asm volatile("vmsgtu.vx v0, v1, %0" :: "r"(two));  /* bits 2..7 set */

	asm volatile("vid.v v2");
	asm volatile("vse32.v v2, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[7]);

	asm volatile("viota.m v3, v0");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[2]); put(mout[3]); put(mout[7]);

	asm volatile("vmv.v.i v4, 0");
	asm volatile("vcompress.vm v4, v1, v0");
	asm volatile("vse32.v v4, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[5]);

	/* Each scan is read back through a zeroed register, so only the bits
	   it writes are visible. */
	asm volatile("vmv.v.i v5, 0"); asm volatile("vmsbf.m v5, v0");
	asm volatile("vmv.x.s %0, v5" : "=r"(v)); put(v);
	asm volatile("vmv.v.i v6, 0"); asm volatile("vmsof.m v6, v0");
	asm volatile("vmv.x.s %0, v6" : "=r"(v)); put(v);
	asm volatile("vmv.v.i v7, 0"); asm volatile("vmsif.m v7, v0");
	asm volatile("vmv.x.s %0, v7" : "=r"(v)); put(v);
}

/* 15. The addressing modes beyond unit-stride: strided, indexed,
   segmented, whole-register, the mask transfers and fault-only-first. */
static void t_addressing(void)
{
	static const unsigned int src16[16] =
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	static const unsigned int pairs[16] =
		{10,11,20,21,30,31,40,41,50,51,60,61,70,71,80,81};
	static const unsigned int ix[8] = {28,24,20,16,12,8,4,0}; /* byte offsets */
	static const unsigned char mbits[8] = {0xA5,0,0,0,0,0,0,0};
	static unsigned char mask_out[8];
	static unsigned int dst16[16];
	ulong v, stride = 8;
	int i;

	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");

	/* Strided load: every other word. */
	asm volatile("vlse32.v v1, (%0), %1" :: "r"(src16), "r"(stride));
	asm volatile("vse32.v v1, (%0)" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[1]); put(dst16[7]);

	/* Strided store scatters the same vector back out. */
	for (i = 0; i < 16; i++) dst16[i] = 0xFF;
	asm volatile("vsse32.v v1, (%0), %1" :: "r"(dst16), "r"(stride) : "memory");
	put(dst16[0]); put(dst16[1]); put(dst16[2]);

	/* Indexed load, whose offsets reverse the source. */
	asm volatile("vle32.v v2, (%0)" :: "r"(ix));
	asm volatile("vluxei32.v v3, (%0), v2" :: "r"(src16));
	asm volatile("vse32.v v3, (%0)" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[7]);

	/* Indexed store, which reverses it back. */
	for (i = 0; i < 16; i++) dst16[i] = 0xEE;
	asm volatile("vsoxei32.v v3, (%0), v2" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[7]);

	/* Two-field segments: the load deinterleaves, the store reinterleaves. */
	asm volatile("vlseg2e32.v v4, (%0)" :: "r"(pairs));
	asm volatile("vse32.v v4, (%0)" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[7]);
	asm volatile("vse32.v v5, (%0)" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[7]);
	asm volatile("vmv1r.v v6, v4");
	asm volatile("vmv1r.v v4, v5");
	asm volatile("vmv1r.v v5, v6");
	asm volatile("vsseg2e32.v v4, (%0)" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[1]);

	/* Whole-register transfers ignore vl, so both are done at vl=1. */
	asm volatile("vle32.v v7, (%0)" :: "r"(src16));
	asm volatile("vsetivli zero, 1, e32, m1, ta, ma");
	asm volatile("vs1r.v v7, (%0)" :: "r"(dst16) : "memory");
	asm volatile("vl1re32.v v8, (%0)" :: "r"(src16 + 8));
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	put(dst16[0]); put(dst16[7]);
	asm volatile("vse32.v v8, (%0)" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[7]);

	/* The mask transfers move ceil(vl/8) bytes. */
	asm volatile("vlm.v v9, (%0)" :: "r"(mbits));
	asm volatile("vsm.v v9, (%0)" :: "r"(mask_out) : "memory");
	put(mask_out[0]);

	/* Fault-only-first, with nothing to fault on: a plain load that
	   must leave vl alone. */
	asm volatile("vle32ff.v v10, (%0)" :: "r"(src16));
	asm volatile("vse32.v v10, (%0)" :: "r"(dst16) : "memory");
	put(dst16[0]); put(dst16[7]);
	asm volatile("csrr %0, vl" : "=r"(v)); put(v);
}

/* 16. Sign injection, classification, the widening float arithmetic and
   the two reciprocal estimates, whose results the ISA pins down exactly. */
static void t_float2(void)
{
	static const float fs[8] =
		{1.5f, -2.5f, 3.0f, -4.0f, 0.0f, -0.0f, 8.0f, -16.0f};
	static const float fneg[8] =
		{-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
	static const float fone[8] = {1.0f, 2.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	static double dd[8];

	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	asm volatile("vle32.v v1, (%0)" :: "r"(fs));
	asm volatile("vle32.v v2, (%0)" :: "r"(fneg));

	/* vfsgnjx of a value with itself is fabs: the two signs cancel. */
	asm volatile("vfsgnjx.vv v3, v1, v1");
	asm volatile("vse32.v v3, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]);
	asm volatile("vfsgnj.vv v4, v1, v2");
	asm volatile("vse32.v v4, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]);
	asm volatile("vfsgnjn.vv v5, v1, v2");
	asm volatile("vse32.v v5, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]);

	/* vfclass writes a ten-bit value per element, not a mask. */
	asm volatile("vfclass.v v6, v1");
	asm volatile("vse32.v v6, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]); put(mout[4]); put(mout[5]);

	/* Widening float: the sources are single precision, the result double,
	   so the destination is an aligned pair. */
	asm volatile("vfwadd.vv v8, v1, v1");
	asm volatile("vsetivli zero, 8, e64, m2, ta, ma");
	asm volatile("vse64.v v8, (%0)" :: "r"(dd) : "memory");
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	put((ulong)dd[6]);                       /* 8 + 8 */
	asm volatile("vfwmul.vv v10, v1, v1");
	asm volatile("vsetivli zero, 8, e64, m2, ta, ma");
	asm volatile("vse64.v v10, (%0)" :: "r"(dd) : "memory");
	asm volatile("vsetivli zero, 8, e32, m1, ta, ma");
	put((ulong)dd[6]);                       /* 8 * 8 */
	/* and back down again */
	asm volatile("vfncvt.f.f.w v12, v10");
	asm volatile("vse32.v v12, (%0)" :: "r"(mout) : "memory");
	put(mout[6]);

	/* The estimates are defined by table lookup, so these are exact. */
	asm volatile("vle32.v v13, (%0)" :: "r"(fone));
	asm volatile("vfrec7.v v14, v13");
	asm volatile("vse32.v v14, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[1]);
	asm volatile("vfrsqrt7.v v15, v13");
	asm volatile("vse32.v v15, (%0)" :: "r"(mout) : "memory");
	put(mout[0]); put(mout[2]);
}

void _start(void)
{
	t_vsetvli();
	t_int_ops();
	t_masked();
	t_widths();
	t_reduce_slide();
	t_compares();
	t_lmul2();
	t_float();
	t_misc();
	t_vlsu_edges();
	t_csrs();
	t_multiply();
	t_divide();
	t_permute2();
	t_addressing();
	t_float2();
	/* Exit with the address of the results (no setup_linux here, so the
	 * host can't rely on anything else being set up). */
	register volatile ulong *a0 asm("a0") = out;
	asm volatile("li a7, 93; ecall" :: "r"(a0));
	__builtin_unreachable();
}
"#;

fn machine(binary: &[u8], jit: bool) -> Machine {
    // The freestanding toolchain links a single RWX segment.
    let options = MachineOptions {
        memory_max: 8 << 20,
        allow_write_exec_segment: true,
        jit,
        jit_background: false,
        jit_cache: false,
        ..Default::default()
    };
    let mut m = Machine::new(binary.into(), options).unwrap();
    m.setup_minimal_syscalls();
    m
}

fn results(m: &Machine) -> Vec<u64> {
    let out = m.return_value();
    assert_eq!(out, m.address_of("out").unwrap());
    (0..RESULTS)
        .map(|i| m.memory.read_u64(out + 8 * i).unwrap())
        .collect()
}

fn f32b(value: f32) -> u64 {
    u64::from(value.to_bits())
}

fn build() -> Option<Vec<u8>> {
    if !testutil::need_compiler() {
        return None;
    }
    Some(testutil::build_and_load(GUEST, ARGS).unwrap())
}

#[test]
fn rvv_basics() {
    let Some(binary) = build() else { return };
    // The block-based simulator reports the instruction at the end of a
    // block on exceptions; the precise one pinpoints the faulting one.
    let mut m = machine(&binary, false);
    m.set_max_instructions(MAX_INSTRUCTIONS);
    if let Err(e) = m.simulate_precise() {
        let pc = m.cpu.pc();
        panic!(
            "exception at pc={pc:#x}: {e} (instr {:#010x})",
            m.memory.read_u32(pc).unwrap_or(0)
        );
    }
    assert!(!m.instruction_limit_reached());
    let r = results(&m);

    let expected: Vec<u64> = vec![
        // t_vsetvli: VLMAX per SEW/LMUL with VLEN=256
        8,  // e32 m1
        3,  // clamped to avl
        32, // e8 m1
        8,  // e64 m2
        4,  // e32 mf2
        64, // e16 m4
        1,  // e64 mf8: max(1, 256/64/8)
        16, // e16 m1 VLMAX
        5,  // vsetivli 5
        8,  // vsetivli 31 clamps to 8
        32, // vsetvl, vtype=0 (e8 m1)
        0,  // vill: vlmul=100 reserved
        // t_int_ops
        11,                    // vadd[0] = 1+10
        9,                     // vsub[0] = 10-1
        198,                   // vsub[1] = 200-2
        4294967295 - 8,        // vsub[7]
        0,                     // vand[0] = 1&10
        11,                    // vor[0] = 1|10
        2 ^ 200,               // vxor[1]
        1,                     // vminu[0]
        16,                    // vsll[0] = 1<<4
        1,                     // vsrl[0] = 10>>3
        14,                    // vrsub[0] = 15-1
        0xFFFF_FFFF_FFFF_FFFA, // vadd.vi -7 wraps, vmv.x.s sign-extends
        7 + 70,                // vadd[6] via vrgather
        // t_masked
        11,       // active element: 1+10
        0xBEEF,   // masked-off, undisturbed
        1,        // vmerge: mask=1 -> vs1
        50,       // vmerge: mask=0 -> vs2[4]
        4 + 4000, // vl=4 body
        0x1234,   // tail-undisturbed
        // t_widths
        2,             // e8: 1+1
        32,            // e8: bsrc[30]+1
        100 * 100,     // e16
        0x1000,        // e16: 1600*1600 truncated to 16 bits
        0x2_0000_0000, // e64
        2 * 0xdead_beef_cafe,
        // t_reduce_slide
        1 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, // vredsum
        u64::MAX,                          // vredmaxu, vmv.x.s sign-extends
        4,                                 // slidedown 3: src[3]
        8,                                 // dest[4] = src[7]
        0,                                 // dest[5]: src[8] beyond VLMAX -> 0
        2,                                 // slide1down: dest[0] = src[1]
        77,                                // scalar in last element
        100,                               // slideup: dest[0] undisturbed
        100,                               // slideup: dest[1] undisturbed
        1,                                 // dest[2] = src[0]
        2,                                 // dest[3] = src[1]
        6,                                 // vrgather.vi 5
        4,                                 // vrgather.vx 3
        // t_compares
        0x10, // vmseq.vi 5 -> bit 4
        0xFF, // vmand of vmsne: 8 mask bits set
        1,    // vcpop
        4,    // vfirst
        // t_lmul2
        17, // element 8: 9+8
        31, // element 15: 16+15
        // t_float
        f32b(1.0 + 0.5),       // vfadd
        f32b(1.0 - 0.5),       // vfsub
        f32b(8.0 * 0.5),       // vfmul
        36,                    // vfredusum
        f32b(1.0 * 0.5 + 1.0), // vfmacc
        f32b(8.0 * 0.5 + 1.0),
        0,                // vfnmacc: -(+0 * -0) - +0 = +0
        0,                // vfnmadd: -(+0 * -0) - +0 = +0
        f32b(1.0 + 10.0), // vfadd.vf
        1,                // vfcvt.rtz 1.5
        8,                // vfcvt.rtz 8.5
        // t_misc
        50,       // vmv.v.i + vadd.vx
        u64::MAX, // vsaddu saturates, vmv.x.s sign-extends
        1,        // vslide1up scalar
        10,       // vslide1up shifts up
        8,        // vmv1r.v copied all 8 elements
        600000,   // vmv2r.v copied the second register
        // t_vlsu_edges
        1,               // unaligned lane round-trip, element 0
        8,               // unaligned lane round-trip, element 7
        4000,            // vl=4 load, last active element
        0xABCD,          // vl=4 load, tail undisturbed
        4,               // vl=4 store wrote the 4th word
        0x5A5A,          // vl=4 store left the 5th word alone
        10,              // masked load, active element
        0xABCD,          // masked load, inactive element
        1,               // masked store, active element
        0x99,            // masked store, inactive element
        f32b(4.0 + 0.5), // vfadd at vl=4, last active
        f32b(9.0),       // vfadd at vl=4, tail undisturbed
        f32b(2.0),       // masked vfmul.vf, active element
        f32b(9.0),       // masked vfmul.vf, inactive element
        8,               // e64 vfadd
        16,              // e64 vfmul
        // t_csrs
        7,    // vl after vsetivli 7
        0xD0, // vtype: e32 (vsew=2), m1, ta, ma
        32,   // vlenb: VLEN=256 bits
        3,    // vxrm after csrwi 3
        6,    // vcsr packs vxrm in bits 2:1, vxsat in bit 0
        0,    // vstart: this emulator never traps mid-instruction
        1,    // vill set by the reserved vlmul
        0,    // ... which also zeroes vl
        // t_multiply: vd=100, vs1=5, vs2=2 for the accumulating forms
        10,         // vmul   2*5
        21,         // vmul   -3*-7
        0,          // vmul   2^30*4 keeps the low 32 bits
        42,         // vmul   6*7
        0,          // vmulh  high half of 21
        1,          // vmulh  high half of 2^32
        4294967286, // vmulhu both operands unsigned
        4294967293, // vmulhsu vs2 signed, vs1 unsigned
        110,        // vmacc   5*2 + 100
        90,         // vnmsac  100 - 5*2
        502,        // vmadd   5*100 + 2
        4294966798, // vnmsub  2 - 5*100
        // t_divide
        14,         // vdivu 100/7
        613566742,  // vdivu (2^32-100)/7
        4294967295, // vdivu by zero: all ones
        0,          // vdivu 2^31 / (2^32-1)
        14,         // vdiv 100/7
        4294967282, // vdiv -100/7 truncates toward zero
        4294967295, // vdiv by zero: -1
        2147483648, // vdiv INT_MIN/-1 wraps
        2,          // vremu (2^32-100)%7
        7,          // vremu by zero: the dividend
        4294967294, // vrem -100%7 = -2
        7,          // vrem by zero: the dividend
        0,          // vrem INT_MIN%-1
        // t_permute2: the mask has bits 2..7 set
        0, // vid[0]
        7, // vid[7]
        0, // viota[0]: no set bits below it
        0, // viota[2]: still none
        1, // viota[3]: bit 2 counted
        5, // viota[7]
        3, // vcompress packs element 2 down to 0
        8, // ... and element 7 to 5
        3, // vmsbf: bits 0,1 -- before the first set bit
        4, // vmsof: bit 2 alone
        7, // vmsif: bits 0,1,2 -- up to and including it
        // t_addressing
        0, // vlse32 stride 8: elements 0, 2, .. 14
        2,
        14,
        0,    // vsse32 writes every other word ...
        0xFF, // ... leaving the ones between alone
        2,
        7, // vluxei32 with reversing offsets
        0,
        0, // vsoxei32 reverses it back
        7,
        10, // vlseg2e32 field 0
        80,
        11, // vlseg2e32 field 1
        81,
        11, // vsseg2e32 with the fields swapped
        10,
        0, // vs1r.v moved the register despite vl=1
        7,
        8, // vl1re32.v likewise
        15,
        0xA5, // vlm.v/vsm.v round-trip one mask byte
        0,    // vle32ff.v with nothing to fault on
        7,
        8, // ... and vl is left alone
        // t_float2
        f32b(1.5), // vfsgnjx v,v clears the sign
        f32b(2.5),
        f32b(-1.5), // vfsgnj takes the negative sign
        f32b(-2.5),
        f32b(1.5), // vfsgnjn takes its complement
        f32b(2.5),
        1 << 6,      // vfclass +normal
        1 << 1,      // vfclass -normal
        1 << 4,      // vfclass +0
        1 << 3,      // vfclass -0
        16,          // vfwadd 8.0 + 8.0
        64,          // vfwmul 8.0 * 8.0
        f32b(64.0),  // vfncvt.f.f.w back to single
        0x3F7F_0000, // vfrec7(1.0): the 7-bit estimate
        0x3EFF_0000, // vfrec7(2.0)
        0x3F7F_0000, // vfrsqrt7(1.0)
        0x3EFF_0000, // vfrsqrt7(4.0)
    ];
    assert!(expected.len() as u64 <= RESULTS);
    for (n, (got, want)) in r.iter().zip(&expected).enumerate() {
        assert_eq!(got, want, "result {n}: {got:#x} != {want:#x}");
    }
}

/// The same guest through the block-based simulator must produce the same
/// results: the two paths share handlers, but decoding and exception
/// reporting differ. So must the JIT, which leaves the vector instructions
/// to the interpreter.
#[test]
fn rvv_fast_path_parity() {
    let Some(binary) = build() else { return };
    let mut precise = machine(&binary, false);
    precise.set_max_instructions(MAX_INSTRUCTIONS);
    precise.simulate_precise().unwrap();
    assert!(!precise.instruction_limit_reached());

    let mut configs = vec![false];
    if cfg!(feature = "jit") {
        configs.push(true);
    }
    for jit in configs {
        let mut fast = machine(&binary, jit);
        fast.simulate(MAX_INSTRUCTIONS).unwrap();
        assert!(!fast.instruction_limit_reached());
        assert_eq!(results(&fast), results(&precise), "jit {jit}");
        assert_eq!(
            fast.instruction_counter(),
            precise.instruction_counter(),
            "jit {jit}"
        );
    }
}
