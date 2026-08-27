#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

generated=$(mktemp)
trap 'rm -f "$generated"' EXIT
python3 scripts/generate_wire_vectors.py >"$generated"
diff -u reference/vectors/production_v1.sha256 "$generated"
echo "wire vectors ok: deterministic fixture and base-plus-seven delta digests match"
