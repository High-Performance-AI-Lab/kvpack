#!/usr/bin/env python3
"""Generate or verify the checksum-pinned third-party license inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tarfile
import tomllib

from sync_local_registry import CRATES_IO_SOURCE, cached_archive, cargo_home


ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "Cargo.lock"
OUTPUT = ROOT / "THIRD_PARTY_NOTICES.md"
DISALLOWED_LICENSE_MARKERS = ("AGPL", "BUSL", "GPL", "LGPL", "MPL", "SSPL")


def manifest_and_members(archive: Path) -> tuple[dict[str, object], set[str]]:
    with tarfile.open(archive, "r:gz") as crate:
        members = {member.name for member in crate.getmembers()}
        manifests = [name for name in members if name.endswith("/Cargo.toml")]
        if len(manifests) != 1:
            raise ValueError(f"{archive.name} has no unique normalized manifest")
        extracted = crate.extractfile(manifests[0])
        if extracted is None:
            raise ValueError(f"cannot read {archive.name} manifest")
        return tomllib.loads(extracted.read().decode("utf-8")), members


def validate_license_label(label: str, source: str) -> str:
    alternatives = label.upper().replace("(", "").replace(")", "").split(" OR ")
    if alternatives and all(
        any(marker in alternative for marker in DISALLOWED_LICENSE_MARKERS)
        for alternative in alternatives
    ):
        raise ValueError(f"{source} has unapproved license expression {label!r}")
    return label


def license_label(
    archive: Path, manifest: dict[str, object], members: set[str]
) -> str:
    package = manifest.get("package")
    if not isinstance(package, dict):
        raise ValueError(f"{archive.name} has no package table")
    expression = package.get("license")
    if isinstance(expression, str) and expression.strip():
        label = expression.strip()
    else:
        license_file = package.get("license-file")
        if not isinstance(license_file, str) or not license_file.strip():
            raise ValueError(f"{archive.name} declares neither license nor license-file")
        suffix = f"/{license_file.strip()}"
        if not any(name.endswith(suffix) for name in members):
            raise ValueError(f"{archive.name} omits declared license file {license_file}")
        label = f"license-file: {license_file.strip()}"
    return validate_license_label(label, archive.name)


def render() -> str:
    document = tomllib.loads(LOCK.read_text())
    packages = document.get("package")
    if not isinstance(packages, list):
        raise ValueError("Cargo.lock has no package inventory")
    cache_root = cargo_home() / "registry" / "cache"
    rows: list[tuple[str, str, str, str]] = []
    for package in packages:
        if not isinstance(package, dict) or package.get("source") is None:
            continue
        source = package.get("source")
        if source != CRATES_IO_SOURCE:
            raise ValueError(f"unsupported dependency source: {source!r}")
        name = str(package["name"])
        version = str(package["version"])
        expected = package.get("checksum")
        if not isinstance(expected, str):
            raise ValueError(f"{name} {version} has no locked checksum")
        archive = cached_archive(cache_root, f"{name}-{version}.crate")
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"cached archive checksum mismatch for {name} {version}")
        manifest, members = manifest_and_members(archive)
        rows.append((name, version, license_label(archive, manifest, members), expected))
    rows.sort(key=lambda row: (row[0].lower(), row[1]))

    lines = [
        "# Third-party notices",
        "",
        "This inventory is generated from the exact crates.io archives pinned in",
        "`Cargo.lock`. Each archive SHA-256 is verified before its declared license",
        "is recorded. The corresponding license texts remain in each source archive;",
        "redistributors must preserve any package-specific attribution or notice files.",
        "",
        "Run `python3 scripts/generate_third_party_notices.py --check` to verify",
        "the tracked inventory without contacting the network.",
        "",
        "| Package | Version | Declared license | Locked source identity |",
        "| --- | --- | --- | --- |",
    ]
    for name, version, license_value, checksum in rows:
        escaped = license_value.replace("|", "\\|")
        lines.append(f"| `{name}` | `{version}` | `{escaped}` | `{checksum}` |")
    lines.extend(["", f"Inventory rows: {len(rows)}", ""])
    return "\n".join(lines)


def main() -> int:
    expected = render()
    if sys.argv[1:] == ["--check"]:
        if not OUTPUT.exists() or OUTPUT.read_text() != expected:
            print("THIRD_PARTY_NOTICES.md is stale; regenerate it", file=sys.stderr)
            return 1
        print("third-party notice inventory is current")
        return 0
    if sys.argv[1:]:
        print("usage: generate_third_party_notices.py [--check]", file=sys.stderr)
        return 2
    OUTPUT.write_text(expected)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
