#!/bin/sh
# LOC guard: fail on any source file over the hard limit, warn over the
# preferred limit. Keeps files modular before they become a ball of mud.
# Run from anywhere; scans crates/ Rust sources (excludes tests/ and target/).
set -eu
cd "$(dirname "$0")/.."

HARD=1000
WARN=500
fail=0
warned=0

# Our source files only: crate src/, not tests/ (tests may legitimately be
# long) and not vendored reference/ or c-ref/.
files=$(find crates -name '*.rs' -not -path '*/target/*' -not -path '*/tests/*')

for f in $files; do
    n=$(wc -l < "$f" | tr -d ' ')
    if [ "$n" -gt "$HARD" ]; then
        echo "FAIL  $n  $f  (> $HARD hard limit — split into a submodule dir)"
        fail=1
    elif [ "$n" -gt "$WARN" ]; then
        echo "warn  $n  $f  (> $WARN preferred — consider splitting)"
        warned=$((warned + 1))
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "loc_guard: FAILED — a source file exceeds the $HARD-line hard limit."
    exit 1
fi
if [ "$warned" -ne 0 ]; then
    echo "loc_guard: ok (no hard violations; $warned file(s) over the $WARN-line preference)."
else
    echo "loc_guard: ok (all source files <= $WARN lines)."
fi
