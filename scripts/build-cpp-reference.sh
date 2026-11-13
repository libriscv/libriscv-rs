#!/usr/bin/env bash
# Build the C++ libriscv emulator in each interpreter dispatch mode, for
# differential testing and benchmarking against the Rust port:
#
#   target/cpp-ref/threaded/rvlinux   computed goto (C++ default, fastest interpreter)
#   target/cpp-ref/tailcall/rvlinux   clang musttail + preserve_none (what the Rust port mirrors)
#   target/cpp-ref/switch/rvlinux     plain switch (baseline)
#   target/cpp-ref/asmjit/rvlinux     threaded + asmjit native code (only when asked for:
#                                     fetches asmjit unless RISCV_ASMJIT_SOURCE_DIR is set)
#
# Binary translation is disabled in all of them. Run with --accurate for
# instruction counts.
#
#   scripts/build-cpp-reference.sh [threaded|tailcall|switch|asmjit ...]
#
# Env: LIBRISCV_CPP (default ../libriscv), CLANGXX (default: newest clang++-NN found).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPP="${LIBRISCV_CPP:-$ROOT/../libriscv}"
OUT="${CARGO_TARGET_DIR:-$ROOT/target}/cpp-ref"
MODES=("${@:-threaded tailcall switch}")
read -r -a MODES <<<"${MODES[*]}"

find_clang() {
	if [ -n "${CLANGXX:-}" ]; then echo "$CLANGXX"; return; fi
	for v in 23 22 21 20 19; do
		if command -v "clang++-$v" >/dev/null; then echo "clang++-$v"; return; fi
	done
	echo clang++
}

GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)

for mode in "${MODES[@]}"; do
	flags=(-DCMAKE_BUILD_TYPE=Release -DRISCV_BINARY_TRANSLATION=OFF)
	expect=""
	case "$mode" in
		threaded) flags+=(-DRISCV_THREADED=ON -DRISCV_TAILCALL_DISPATCH=OFF); expect="Threaded dispatch enabled" ;;
		tailcall)
			# Needs clang >= 19 for preserve_none; CMake silently falls back
			# to threaded dispatch otherwise, so we verify below.
			flags+=(-DRISCV_TAILCALL_DISPATCH=ON -DCMAKE_CXX_COMPILER="$(find_clang)")
			expect="Tail-call dispatch enabled" ;;
		switch) flags+=(-DRISCV_THREADED=OFF -DRISCV_TAILCALL_DISPATCH=OFF); expect="Switch-based dispatch enabled" ;;
		asmjit)
			flags+=(-DRISCV_THREADED=ON -DRISCV_TAILCALL_DISPATCH=OFF -DRISCV_LIBTCC=OFF -DRISCV_ASMJIT=ON)
			[ -n "${RISCV_ASMJIT_SOURCE_DIR:-}" ] && flags+=(-DRISCV_ASMJIT_SOURCE_DIR="$RISCV_ASMJIT_SOURCE_DIR")
			expect="Threaded dispatch enabled" ;;
		*) echo "Unknown mode: $mode (threaded|tailcall|switch|asmjit)" >&2; exit 2 ;;
	esac
	build="$OUT/$mode"
	echo "==> $mode"
	mkdir -p "$build"
	rm -f "$build/CMakeCache.txt"
	cmake -S "$CPP/emulator" -B "$build" "${GEN[@]}" "${flags[@]}" >"$build.cmake.log" 2>&1 \
		|| { cat "$build.cmake.log"; exit 1; }
	if ! grep -q "$expect" "$build.cmake.log"; then
		echo "error: expected '$expect' in CMake output for $mode:" >&2
		grep 'libriscv:' "$build.cmake.log" >&2 || true
		exit 1
	fi
	cmake --build "$build" --target rvlinux >/dev/null
	echo "    $build/rvlinux"
done
