#!/usr/bin/env python3
"""Inject and verify complete, byte-exact notices in release archives."""

from __future__ import annotations

import argparse
import copy
import gzip
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import tomllib

from release_packages import EXPECTED_INTERNAL_DEPENDENCIES, PACKAGES, VERSION


ROOT = Path(__file__).resolve().parent.parent
NOTICE_NAMES = ("LICENSE-APACHE", "LICENSE-MIT", "THIRD_PARTY_NOTICES.md")
NOTICE_BYTES = {name: (ROOT / name).read_bytes() for name in NOTICE_NAMES}


def expected_archives(directory: Path) -> list[Path]:
    paths = [directory / f"{package}-{VERSION}.crate" for package in PACKAGES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"required package archives are absent: {missing!r}")
    return paths


def safe_member(member: tarfile.TarInfo, expected_root: str) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member: {member.name!r}")
    if path.parts[0] != expected_root:
        raise ValueError(f"archive member escapes package root: {member.name!r}")
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f"links and device members are forbidden: {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"unsupported archive member type: {member.name!r}")


def package_identity(archive: Path) -> tuple[str, str]:
    suffix = f"-{VERSION}.crate"
    if not archive.name.endswith(suffix):
        raise ValueError(f"unexpected package archive name: {archive.name}")
    package = archive.name[: -len(suffix)]
    if package not in PACKAGES:
        raise ValueError(f"archive is not a current release package: {archive.name}")
    return package, f"{package}-{VERSION}"


def read_archive(archive: Path) -> tuple[str, list[tuple[tarfile.TarInfo, bytes | None]]]:
    _, package_root = package_identity(archive)
    records: list[tuple[tarfile.TarInfo, bytes | None]] = []
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            safe_member(member, package_root)
            if member.name in seen:
                raise ValueError(f"duplicate archive member: {member.name!r}")
            seen.add(member.name)
            data: bytes | None = None
            if member.isfile():
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                data = extracted.read()
                if len(data) != member.size:
                    raise ValueError(f"truncated archive member: {member.name!r}")
            records.append((copy.copy(member), data))
    for required in ("Cargo.toml", "Cargo.toml.orig", "README.md"):
        if f"{package_root}/{required}" not in seen:
            raise ValueError(f"archive omits required package file: {required}")
    return package_root, records


def inject(archive: Path) -> None:
    package_root, records = read_archive(archive)
    by_name = {member.name: data for member, data in records}
    changed = False
    for name, expected in NOTICE_BYTES.items():
        member_name = f"{package_root}/{name}"
        actual = by_name.get(member_name)
        if actual is not None:
            if actual != expected:
                raise ValueError(f"archive contains a noncanonical {name}")
            continue
        member = tarfile.TarInfo(member_name)
        member.size = len(expected)
        member.mode = 0o644
        member.mtime = 0
        member.uid = 0
        member.gid = 0
        member.uname = ""
        member.gname = ""
        records.append((member, expected))
        changed = True
    if not changed:
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as encoded:
                with tarfile.open(fileobj=encoded, mode="w", format=tarfile.PAX_FORMAT) as target:
                    for member, data in records:
                        target.addfile(member, None if data is None else io.BytesIO(data))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            temporary.unlink()


def check(archive: Path) -> dict[str, object]:
    package_root, records = read_archive(archive)
    by_name = {member.name: data for member, data in records}
    for name, expected in NOTICE_BYTES.items():
        member_name = f"{package_root}/{name}"
        if by_name.get(member_name) != expected:
            raise ValueError(f"{archive.name} omits byte-exact {name}")
    manifest_bytes = by_name.get(f"{package_root}/Cargo.toml")
    if manifest_bytes is None:
        raise ValueError(f"{archive.name} omits its normalized manifest")
    manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
    package = manifest.get("package")
    if not isinstance(package, dict):
        raise ValueError(f"{archive.name} has no package metadata")
    expected_name = package_root[: -len(f"-{VERSION}")]
    expected_metadata = {
        "name": expected_name,
        "version": VERSION,
        "license": "MIT OR Apache-2.0",
        "repository": "https://github.com/High-Performance-AI-Lab/kvpack",
        "readme": "README.md",
        "rust-version": "1.85",
    }
    for field, expected in expected_metadata.items():
        if package.get(field) != expected:
            raise ValueError(
                f"{archive.name} metadata {field!r} must be {expected!r}, "
                f"got {package.get(field)!r}"
            )
    check_normalized_dependencies(archive, manifest)
    return {
        "archive": archive.name,
        "notices": list(NOTICE_NAMES),
        "members": len(records),
        "metadata": expected_metadata,
    }


def check_normalized_dependencies(archive: Path, document: dict[str, object]) -> None:
    package_name, _ = package_identity(archive)
    observed_internal: set[str] = set()
    tables: list[tuple[str, object]] = [
        (name, document.get(name, {}))
        for name in ("dependencies", "dev-dependencies", "build-dependencies")
    ]
    targets = document.get("target", {})
    if not isinstance(targets, dict):
        raise ValueError(f"{archive.name} has an invalid target table")
    for target, target_document in targets.items():
        if not isinstance(target_document, dict):
            raise ValueError(f"{archive.name} has an invalid target {target!r} table")
        tables.extend(
            (f"target.{target}.{name}", target_document.get(name, {}))
            for name in ("dependencies", "dev-dependencies", "build-dependencies")
        )
    for table_name, table in tables:
        if not isinstance(table, dict):
            raise ValueError(f"{archive.name} has an invalid {table_name} table")
        for name, value in table.items():
            if isinstance(value, str):
                version = value
            elif isinstance(value, dict):
                forbidden = {
                    "branch",
                    "git",
                    "path",
                    "registry",
                    "registry-index",
                    "rev",
                    "tag",
                    "workspace",
                }.intersection(value)
                if forbidden:
                    raise ValueError(
                        f"{archive.name}:{table_name}.{name} retained source keys "
                        f"{sorted(forbidden)!r}"
                    )
                version = value.get("version")
            else:
                raise ValueError(
                    f"{archive.name}:{table_name}.{name} has an invalid dependency"
                )
            if not isinstance(version, str) or not version.startswith("="):
                raise ValueError(
                    f"{archive.name}:{table_name}.{name} is not exact: {version!r}"
                )
            if name in PACKAGES and version != f"={VERSION}":
                raise ValueError(
                    f"{archive.name}:{table_name}.{name} must use '={VERSION}'"
                )
            if name in PACKAGES:
                observed_internal.add(name)
    expected_internal = EXPECTED_INTERNAL_DEPENDENCIES[package_name]
    if observed_internal != expected_internal:
        raise ValueError(
            f"{archive.name} normalized internal dependency set is "
            f"{sorted(observed_internal)!r}, expected {sorted(expected_internal)!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inject", "check"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    path = arguments.path.resolve(strict=True)
    if arguments.mode == "inject":
        if not path.is_file():
            raise ValueError("inject requires one archive file")
        inject(path)
        result = check(path)
        print(f"package notices injected and verified: {result['archive']}")
        return 0
    archives = expected_archives(path) if path.is_dir() else [path]
    results = [check(archive) for archive in archives]
    print(f"package notice policy ok: {len(results)} archives, {len(NOTICE_NAMES)} files each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
