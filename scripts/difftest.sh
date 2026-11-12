#!/usr/bin/env bash
# Run one RISC-V ELF under the C++ reference emulator and the Rust rvlinux,
# and diff guest stdout, exit code and instruction count.
#
#   scripts/difftest.sh prog.elf [guest args...]
set -euo pipefail

if [ $# -lt 1 ]; then
	echo "Usage: $0 <program.elf> [args...]" >&2
	exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPP="${LIBRISCV_CPP:-$ROOT/../libriscv}"
REF="${LIBRISCV_CPP_EMULATOR:-$CPP/emulator/rvlinux}"

cargo build --quiet --release -p rvlinux --manifest-path "$ROOT/Cargo.toml"
RUST="${CARGO_TARGET_DIR:-$ROOT/target}/release/rvlinux"

# Drop timing-dependent lines; keep guest output, exit code and instruction count.
filter() {
	sed -E -e 's/  Runtime: .*$//' -e '/^Runtime: /d' -e '/^Pages in use: /d'
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

"$REF" --no-translate --accurate "$@" </dev/null 2>"$tmp/ref.err" | filter >"$tmp/ref.out" || true
"$RUST" --accurate "$@" </dev/null 2>"$tmp/rust.err" | filter >"$tmp/rust.out" || true

if diff -u --label c++ --label rust "$tmp/ref.out" "$tmp/rust.out"; then
	echo "OK: outputs match"
else
	echo "--- c++ stderr:"; cat "$tmp/ref.err"
	echo "--- rust stderr:"; cat "$tmp/rust.err"
	exit 1
fi
