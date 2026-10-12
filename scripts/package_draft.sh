#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp_base="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
work_dir="$(mktemp -d "$temp_base/kvpack-draft-registry.XXXXXX")"
registry_dir="$work_dir/registry"
cargo_home="$work_dir/cargo-home"
consumer_dir="$work_dir/consumer"
package_target="$work_dir/package-target"
package_repo="$work_dir/package-source"
version="$(python3 "$repo_dir/scripts/release_packages.py" version)"
packages=()
while IFS= read -r package; do
    packages+=("$package")
done < <(python3 "$repo_dir/scripts/release_packages.py" packages)

cleanup() {
    exit_code=$?
    trap - EXIT
    if [[ "$work_dir" == "$temp_base/kvpack-draft-registry."* ]]; then
        rm -rf -- "$work_dir"
    fi
    exit "$exit_code"
}
trap cleanup EXIT

allow_dirty="${KVPACK_PACKAGE_ALLOW_DIRTY:-0}"
if [[ "$allow_dirty" != "0" && "$allow_dirty" != "1" ]]; then
    echo "KVPACK_PACKAGE_ALLOW_DIRTY must be 0 or 1" >&2
    exit 2
fi
dirty="$(git -C "$repo_dir" status --porcelain=v1 --untracked-files=all)"
if [[ -n "$dirty" && "$allow_dirty" != "1" ]]; then
    echo "refusing to package a dirty kvpack worktree" >&2
    exit 1
fi

python3 "$repo_dir/scripts/check_manifest_policy.py"
python3 "$repo_dir/scripts/check_dependency_urls.py"
python3 "$repo_dir/scripts/generate_third_party_notices.py" --check

mkdir -p "$cargo_home" "$consumer_dir/src"
python3 "$repo_dir/scripts/sync_local_registry.py" \
    "$repo_dir/Cargo.lock" \
    "$registry_dir"
sed -e "s|@REGISTRY@|$registry_dir|g" \
    "$repo_dir/scripts/local-registry-config.toml.in" \
    > "$cargo_home/config.toml"

# Package from an isolated copy so Cargo cannot mutate the candidate lock or
# source tree. Dirty copies are allowed only for local pre-commit validation.
rsync -a --exclude .git --exclude target "$repo_dir/" "$package_repo/"
(
    cd "$package_repo"
    CARGO_HOME="$cargo_home" cargo generate-lockfile --offline
)

for package in "${packages[@]}"; do
    package_args=(
        package
        --manifest-path "$package_repo/Cargo.toml"
        --package "$package"
        --locked
        --offline
        --no-verify
        --target-dir "$package_target"
    )
    (
        cd "$package_repo"
        CARGO_HOME="$cargo_home" cargo "${package_args[@]}"
    )
    archive="$package_target/package/${package}-${version}.crate"
    python3 "$repo_dir/scripts/package_archive_policy.py" inject "$archive"
    cp "$archive" "$registry_dir/"
    python3 "$repo_dir/scripts/local_registry_index.py" \
        "$registry_dir" \
        "$registry_dir/${package}-${version}.crate"
done

python3 "$repo_dir/scripts/package_archive_policy.py" check "$registry_dir"

if [[ -n "${KVPACK_PACKAGE_OUTPUT_DIR:-}" ]]; then
    output_dir="$KVPACK_PACKAGE_OUTPUT_DIR"
    if [[ -L "$output_dir" ]]; then
        echo "package output directory must not be a symlink: $output_dir" >&2
        exit 2
    fi
    if [[ -e "$output_dir" && ! -d "$output_dir" ]]; then
        echo "package output path must be a directory: $output_dir" >&2
        exit 2
    fi
    if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "package output directory must be absent or empty: $output_dir" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    for package in "${packages[@]}"; do
        cp "$registry_dir/${package}-${version}.crate" "$output_dir/"
    done
    python3 "$repo_dir/scripts/package_archive_policy.py" check "$output_dir"
    python3 "$repo_dir/scripts/package_archive_digests.py" "$output_dir"
fi

if [[ "${KVPACK_PACKAGE_ARCHIVES_ONLY:-0}" == "1" ]]; then
    echo "draft package archives created"
    exit 0
fi

cp "$repo_dir/scripts/package-consumer.Cargo.toml" "$consumer_dir/Cargo.toml"
cp "$repo_dir/scripts/package-consumer.rs" "$consumer_dir/src/main.rs"
(
    cd "$consumer_dir"
    CARGO_HOME="$cargo_home" CARGO_TARGET_DIR="$work_dir/consumer-target" \
        cargo run --offline
)

mkdir -p "$work_dir/unpacked"
for package in "${packages[@]}"; do
    tar -xzf "$registry_dir/${package}-${version}.crate" -C "$work_dir/unpacked"
    (
        cd "$work_dir/unpacked/${package}-${version}"
        CARGO_HOME="$cargo_home" CARGO_TARGET_DIR="$work_dir/package-check-target" \
            cargo check --all-targets --offline
    )
done

echo "draft registry package check passed"
