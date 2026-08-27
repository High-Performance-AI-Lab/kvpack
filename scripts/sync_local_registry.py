#!/usr/bin/env python3
"""Build a Cargo local registry from the exact archives in a Cargo.lock file.

The script never contacts a registry. It verifies each cached crates.io archive
against the checksum pinned in the lock file before copying and indexing it.
Workspace packages have no registry source and are added later by the packaging
driver after `cargo package` creates their archives.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import tomllib

from local_registry_index import index_archive


CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"


def cargo_home() -> Path:
    configured = os.environ.get("CARGO_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cargo"


def cached_archive(cache_root: Path, filename: str) -> Path:
    matches = sorted(cache_root.glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"locked archive {filename} is absent from {cache_root}; run "
            "`cargo fetch --locked` before the offline package qualification"
        )
    if len(matches) > 1:
        digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in matches}
        if len(digests) != 1:
            raise ValueError(f"cached archives for {filename} disagree")
    return matches[0]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sync_local_registry.py CARGO_LOCK REGISTRY", file=sys.stderr)
        return 2

    lock_path = Path(sys.argv[1]).resolve(strict=True)
    registry = Path(sys.argv[2]).resolve()
    registry.mkdir(parents=True, exist_ok=True)
    cache_root = cargo_home() / "registry" / "cache"
    document = tomllib.loads(lock_path.read_text())
    packages = document.get("package")
    if not isinstance(packages, list):
        raise ValueError("Cargo.lock has no package inventory")

    copied = 0
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("Cargo.lock contains an invalid package row")
        source = package.get("source")
        if source is None:
            continue
        name = str(package["name"])
        version = str(package["version"])
        if source != CRATES_IO_SOURCE:
            raise ValueError(f"unsupported locked package source: {source!r}")
        expected = package.get("checksum")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"locked package {name} {version} has no SHA-256 checksum")
        filename = f"{name}-{version}.crate"
        source_archive = cached_archive(cache_root, filename)
        actual = hashlib.sha256(source_archive.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"cached archive checksum mismatch for {name} {version}: "
                f"expected {expected}, got {actual}"
            )
        destination = registry / filename
        shutil.copyfile(source_archive, destination)
        index_archive(registry, destination)
        copied += 1

    print(f"local registry synchronized: {copied} checksum-pinned archives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
