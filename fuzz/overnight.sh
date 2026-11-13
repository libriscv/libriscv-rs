#!/bin/bash
# Fuzz every target for a long, unattended run (e.g. overnight), splitting
# the cores between them, and summarise the findings at the end.
#
# Unlike do_fuzz.sh, a finding doesn't stop a target: libFuzzer's fork mode
# keeps going after crashes, OOMs and timeouts (-ignore_*), writing each
# reproducer to artifacts/<target>/. Corpora accumulate in corpus/<target>/
# across runs.
#
# Usage:
#   ./overnight.sh [seconds] [targets...]
#
# Examples:
#   ./overnight.sh                      # all targets, 8 hours
#   ./overnight.sh 3600 linuxfuzzer     # one target, one hour
#
# Environment:
#   JOBS          worker processes in total (default: number of cores)
#   FUZZ_FLAGS    cargo-fuzz flags (default "-s none -a"; jitdifffuzzer adds
#                 --features jit). For the JIT's unsafe runtime, consider
#                 FUZZ_FLAGS="-s address -a" on an x86-64 host.
#   LIBRISCV_CPP  C++ libriscv checkout, for the elffuzzer seeds
#   NO_JIT=1      skip jitdifffuzzer (it only compares anything on x86-64)
set -e

SECONDS_TOTAL="${1:-28800}"
shift $(( $# < 1 ? $# : 1 )) || true

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
	TARGETS=(vmfuzzer elffuzzer nativefuzzer syshelperfuzzer linuxfuzzer
		libcfuzzer forkfuzzer guestfuzzer vmcallfuzzer)
	if [ -z "${NO_JIT:-}" ]; then
		TARGETS+=(jitdifffuzzer)
	fi
fi

CORES="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
PER_TARGET=$(( CORES / ${#TARGETS[@]} ))
if [ "$PER_TARGET" -lt 1 ]; then
	PER_TARGET=1
fi

LOGS="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGS"
echo "Fuzzing ${#TARGETS[@]} targets x $PER_TARGET workers for ${SECONDS_TOTAL}s; logs in $LOGS/"

# Build first, so the targets don't race on the build directory.
FLAGS="${FUZZ_FLAGS:--s none -a}"
# shellcheck disable=SC2086
cargo fuzz build $FLAGS
for t in "${TARGETS[@]}"; do
	if [ "$t" = "jitdifffuzzer" ]; then
		# shellcheck disable=SC2086
		cargo fuzz build $FLAGS --features jit --target-dir target/jit jitdifffuzzer
	fi
done

for t in "${TARGETS[@]}"; do
	mkdir -p "artifacts/$t"
	./do_fuzz.sh "$t" -- \
		-fork="$PER_TARGET" \
		-ignore_crashes=1 -ignore_ooms=1 -ignore_timeouts=1 \
		-timeout=30 -rss_limit_mb=2048 \
		-max_total_time="$SECONDS_TOTAL" \
		-artifact_prefix="artifacts/$t/" \
		-print_final_stats=1 \
		>"$LOGS/$t.log" 2>&1 &
	sleep 2
done
wait || true

echo
echo "Findings (artifacts/<target>/):"
for t in "${TARGETS[@]}"; do
	crashes=$(find "artifacts/$t" -name 'crash-*' 2>/dev/null | wc -l | tr -d ' ')
	ooms=$(find "artifacts/$t" -name 'oom-*' 2>/dev/null | wc -l | tr -d ' ')
	timeouts=$(find "artifacts/$t" -name 'timeout-*' 2>/dev/null | wc -l | tr -d ' ')
	cov=$(grep -Eo 'cov: [0-9]+' "$LOGS/$t.log" | tail -1)
	printf '  %-16s crashes %3s  ooms %3s  timeouts %3s  %s\n' \
		"$t" "$crashes" "$ooms" "$timeouts" "$cov"
done
echo
echo "Replay one: cargo fuzz run -s none -a <target> artifacts/<target>/<file>"
