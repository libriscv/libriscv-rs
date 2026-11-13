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
#   FUZZ_FLAGS     cargo-fuzz flags (default: "-s none -a", see README.md)
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

# shellcheck disable=SC2086
exec cargo fuzz run ${FUZZ_FLAGS:--s none -a} "$TARGET" "${CORPUS[@]}" -- "$@"
