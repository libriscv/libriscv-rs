#!/bin/bash
# Run one of the fuzzers (port of libriscv's fuzz/do_fuzz.sh).
#
# Usage:
#   ./do_fuzz.sh [target] [-- extra libFuzzer args]
#
# Examples:
#   ./do_fuzz.sh                                   # vmfuzzer
#   ./do_fuzz.sh elffuzzer -- -max_total_time=300  # seeded ELF loader
#   ./do_fuzz.sh vmfuzzer -- -fork=16              # 16 worker processes
#
# Environment:
#   FUZZ_FLAGS     cargo-fuzz flags (default: "-s none -a", see README.md;
#                  jitdifffuzzer adds --features jit)
#   LIBRISCV_CPP   C++ libriscv checkout, for the ELF seeds (default ../../libriscv)
set -e

TARGET="${1:-vmfuzzer}"
shift $(( $# < 1 ? $# : 1 )) || true
if [ "${1:-}" = "--" ]; then shift; fi

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# libFuzzer shells out to llvm-symbolizer, whose debuginfod client can hang
# forever on distros with a default debuginfod URL (Ubuntu). See README.md.
export DEBUGINFOD_URLS=

CORPUS=()
if [ "$TARGET" = "elffuzzer" ]; then
	# An empty corpus never gets past ELF header validation: seed with real
	# RV64 binaries. The writable corpus goes first, so libFuzzer never
	# writes into the seed directory.
	CPP="${LIBRISCV_CPP:-$HERE/../../libriscv}"
	mkdir -p corpus/elffuzzer seeds/elffuzzer
	for f in "$CPP"/tests/unit/elf/*; do
		if [ -f "$f" ] && head -c 5 "$f" | cmp -s - <(printf '\x7fELF\x02'); then
			cp -u "$f" seeds/elffuzzer/
		fi
	done
	CORPUS=(corpus/elffuzzer seeds/elffuzzer)
fi

FLAGS="${FUZZ_FLAGS:--s none -a}"
if [ "$TARGET" = "jitdifffuzzer" ] && [[ "$FLAGS" != *jit* ]]; then
	# Its own target directory: rebuilding the shared one with the JIT
	# would switch every other target to the JIT too (fork mode re-runs
	# the binary for each job).
	FLAGS="$FLAGS --features jit --target-dir target/jit"
fi

# Instruction words, system call numbers, sandbox paths and guest pointers
# the target needs (see dicts/). Passed first, so a later -dict= wins.
DICT=()
if [ -f "dicts/$TARGET.dict" ]; then
	DICT=("-dict=dicts/$TARGET.dict")
fi

# shellcheck disable=SC2086
exec cargo fuzz run $FLAGS "$TARGET" "${CORPUS[@]}" -- "${DICT[@]}" "$@"
