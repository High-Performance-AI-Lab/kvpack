#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
first="$(mktemp -d "${TMPDIR:-/tmp}/kvpack-package-first.XXXXXX")"
second="$(mktemp -d "${TMPDIR:-/tmp}/kvpack-package-second.XXXXXX")"

cleanup() {
    exit_code=$?
    trap - EXIT
    for directory in "$first" "$second"; do
        if [[ "$directory" == "${TMPDIR:-/tmp}/kvpack-package-"* ]]; then
            rm -rf -- "$directory"
        fi
    done
    exit "$exit_code"
}
trap cleanup EXIT

allow_dirty="${KVPACK_PACKAGE_ALLOW_DIRTY:-0}"
KVPACK_PACKAGE_ALLOW_DIRTY="$allow_dirty" \
KVPACK_PACKAGE_ARCHIVES_ONLY=1 \
KVPACK_PACKAGE_OUTPUT_DIR="$first/output" \
    "$repo_dir/scripts/package_draft.sh"
KVPACK_PACKAGE_ALLOW_DIRTY="$allow_dirty" \
KVPACK_PACKAGE_ARCHIVES_ONLY=1 \
KVPACK_PACKAGE_OUTPUT_DIR="$second/output" \
    "$repo_dir/scripts/package_draft.sh"

diff -u "$first/output/SHA256SUMS" "$second/output/SHA256SUMS"
echo "reproducible package check passed"
