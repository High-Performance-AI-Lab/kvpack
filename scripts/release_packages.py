#!/usr/bin/env python3
"""Canonical package identities for the current kvpack release line."""

from __future__ import annotations

from pathlib import Path
import sys


VERSION = "0.1.0-alpha.2"
PACKAGE_PATHS = {
    "kvpack-core": Path("crates/kvpack-core"),
    "kvpack": Path("crates/kvpack"),
    "kvpack-handoff": Path("crates/kvpack-handoff"),
}
PACKAGES = tuple(PACKAGE_PATHS)
EXPECTED_INTERNAL_DEPENDENCIES = {
    "kvpack-core": set(),
    "kvpack": {"kvpack-core"},
    "kvpack-handoff": set(),
}


def main() -> int:
    if sys.argv[1:] == ["version"]:
        print(VERSION)
        return 0
    if sys.argv[1:] == ["packages"]:
        print("\n".join(PACKAGES))
        return 0
    print("usage: release_packages.py {version|packages}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
