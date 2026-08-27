#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for kvpack package archives."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from release_packages import PACKAGES, VERSION


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: package_archive_digests.py PACKAGE_DIRECTORY", file=sys.stderr)
        return 2
    directory = Path(sys.argv[1]).resolve(strict=True)
    archives = sorted(directory.glob("*.crate"), key=lambda path: path.name)
    expected = sorted(f"{package}-{VERSION}.crate" for package in PACKAGES)
    actual = [archive.name for archive in archives]
    if actual != expected:
        raise ValueError(f"package archive set mismatch: expected {expected!r}, got {actual!r}")
    lines = [
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}"
        for archive in archives
    ]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
