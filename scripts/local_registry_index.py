#!/usr/bin/env python3
"""Add one locally packaged .crate archive to a Cargo local-registry index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tomllib


def index_path(registry: Path, name: str) -> Path:
    lowered = name.lower()
    if len(lowered) == 1:
        relative = Path("1") / lowered
    elif len(lowered) == 2:
        relative = Path("2") / lowered
    elif len(lowered) == 3:
        relative = Path("3") / lowered[0] / lowered
    else:
        relative = Path(lowered[:2]) / lowered[2:4] / lowered
    return registry / "index" / relative


def dependency_rows(
    table: dict[str, object], kind: str | None, target: str | None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for explicit_name, value in table.items():
        if isinstance(value, str):
            spec: dict[str, object] = {"version": value}
        elif isinstance(value, dict):
            spec = value
        else:
            raise ValueError(f"invalid dependency {explicit_name!r}")
        package_name = str(spec.get("package", explicit_name))
        rows.append(
            {
                "name": explicit_name,
                "req": str(spec.get("version", "*")),
                "features": list(spec.get("features", [])),
                "optional": bool(spec.get("optional", False)),
                "default_features": bool(spec.get("default-features", True)),
                "target": target,
                "kind": kind,
                "package": package_name if package_name != explicit_name else None,
            }
        )
    return rows


def manifest_from_archive(archive: Path) -> dict[str, object]:
    with tarfile.open(archive, "r:gz") as crate:
        members = [member for member in crate.getmembers() if member.name.endswith("/Cargo.toml")]
        if len(members) != 1:
            raise ValueError("archive must contain exactly one normalized Cargo.toml")
        extracted = crate.extractfile(members[0])
        if extracted is None:
            raise ValueError("cannot read normalized Cargo.toml")
        return tomllib.loads(extracted.read().decode("utf-8"))


def index_archive(registry: Path, archive: Path) -> None:
    """Add or replace one checked archive record in a local registry."""

    registry = registry.resolve(strict=True)
    archive = archive.resolve(strict=True)
    manifest = manifest_from_archive(archive)
    package = manifest["package"]
    if not isinstance(package, dict):
        raise ValueError("missing package table")
    name = str(package["name"])
    version = str(package["version"])
    expected_name = f"{name}-{version}.crate"
    if archive.name != expected_name or archive.parent != registry:
        raise ValueError("archive name or location does not match its manifest")

    dependencies: list[dict[str, object]] = []
    for table_name, kind in (
        ("dependencies", None),
        ("dev-dependencies", "dev"),
        ("build-dependencies", "build"),
    ):
        table = manifest.get(table_name, {})
        if not isinstance(table, dict):
            raise ValueError(f"invalid {table_name} table")
        dependencies.extend(dependency_rows(table, kind, None))

    targets = manifest.get("target", {})
    if not isinstance(targets, dict):
        raise ValueError("invalid target table")
    for target, tables in targets.items():
        if not isinstance(tables, dict):
            raise ValueError("invalid target dependency table")
        for table_name, kind in (
            ("dependencies", None),
            ("dev-dependencies", "dev"),
            ("build-dependencies", "build"),
        ):
            table = tables.get(table_name, {})
            if not isinstance(table, dict):
                raise ValueError(f"invalid target {table_name} table")
            dependencies.extend(dependency_rows(table, kind, target))

    dependencies.sort(
        key=lambda row: (
            str(row["name"]),
            str(row["kind"]),
            str(row["target"]),
        )
    )
    features = manifest.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("invalid features table")
    record = {
        "name": name,
        "vers": version,
        "deps": dependencies,
        "cksum": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "features": features,
        "yanked": False,
        "links": package.get("links"),
    }

    destination = index_path(registry, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prior: list[dict[str, object]] = []
    if destination.exists():
        prior = [json.loads(line) for line in destination.read_text().splitlines() if line]
    prior = [row for row in prior if row.get("vers") != version]
    prior.append(record)
    prior.sort(key=lambda row: str(row["vers"]))
    destination.write_text(
        "\n".join(json.dumps(row, separators=(",", ":"), sort_keys=True) for row in prior)
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: local_registry_index.py REGISTRY ARCHIVE", file=sys.stderr)
        return 2

    index_archive(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
