#!/usr/bin/env python3
"""Enforce the source-manifest policy for the current release workspace."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib

from release_packages import (
    EXPECTED_INTERNAL_DEPENDENCIES,
    PACKAGE_PATHS,
    PACKAGES,
    VERSION,
)


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_REPOSITORY = "https://github.com/High-Performance-AI-Lab/kvpack"
DEPENDENCY_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
EXACT_SEMVER = re.compile(
    r"^=(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
FORBIDDEN_SOURCE_KEYS = {
    "branch",
    "git",
    "package",
    "registry",
    "registry-index",
    "rev",
    "tag",
}


def manifests() -> tuple[Path, ...]:
    return (ROOT / "Cargo.toml",) + tuple(
        ROOT / PACKAGE_PATHS[package] / "Cargo.toml" for package in PACKAGES
    )


def check_dependency(
    errors: list[str], manifest: Path, table_name: str, name: str, value: object
) -> None:
    label = f"{manifest.relative_to(ROOT)}:{table_name}.{name}"
    if isinstance(value, str):
        if name in PACKAGE_PATHS:
            errors.append(f"{label}: workspace dependency must use its canonical path")
        if EXACT_SEMVER.fullmatch(value) is None:
            errors.append(f"{label}: dependency requirement is not exact: {value!r}")
        return
    if not isinstance(value, dict):
        errors.append(f"{label}: dependency must be a string or table")
        return
    forbidden = FORBIDDEN_SOURCE_KEYS.intersection(value)
    if forbidden:
        errors.append(f"{label}: dependency uses forbidden source keys {sorted(forbidden)!r}")
    path = value.get("path")
    dependency_name = name
    if path is not None:
        if dependency_name not in PACKAGE_PATHS:
            errors.append(f"{label}: only release-workspace dependencies may use a path")
        elif not isinstance(path, str):
            errors.append(f"{label}: dependency path must be a string")
        else:
            expected = (ROOT / PACKAGE_PATHS[dependency_name]).resolve()
            try:
                actual = (manifest.parent / path).resolve(strict=True)
            except OSError:
                errors.append(f"{label}: dependency path is absent")
            else:
                if actual != expected:
                    errors.append(f"{label}: dependency path is not canonical")
            if value.get("version") != f"={VERSION}":
                errors.append(
                    f"{label}: workspace path dependency must use version '={VERSION}'"
                )
    elif dependency_name in PACKAGE_PATHS:
        errors.append(f"{label}: workspace dependency must use its canonical path")
    if value.get("workspace") is True:
        return
    version = value.get("version")
    if not isinstance(version, str) or EXACT_SEMVER.fullmatch(version) is None:
        errors.append(f"{label}: dependency requirement is not exact: {version!r}")


def check_tables(errors: list[str], manifest: Path, document: dict[str, object]) -> None:
    for table_name in DEPENDENCY_TABLES:
        table = document.get(table_name, {})
        if not isinstance(table, dict):
            errors.append(f"{manifest.relative_to(ROOT)}:{table_name}: invalid table")
            continue
        for name, value in table.items():
            check_dependency(errors, manifest, table_name, name, value)

    targets = document.get("target", {})
    if not isinstance(targets, dict):
        errors.append(f"{manifest.relative_to(ROOT)}:target: invalid table")
        return
    for target, tables in targets.items():
        if not isinstance(tables, dict):
            errors.append(f"{manifest.relative_to(ROOT)}:target.{target}: invalid table")
            continue
        check_tables(errors, manifest, tables)


def observed_internal_dependencies(document: dict[str, object]) -> set[str]:
    observed: set[str] = set()
    for table_name in DEPENDENCY_TABLES:
        table = document.get(table_name, {})
        if isinstance(table, dict):
            observed.update(set(table).intersection(PACKAGE_PATHS))
    targets = document.get("target", {})
    if isinstance(targets, dict):
        for target_document in targets.values():
            if isinstance(target_document, dict):
                observed.update(observed_internal_dependencies(target_document))
    return observed


def main() -> int:
    errors: list[str] = []
    for manifest in manifests():
        document = tomllib.loads(manifest.read_text())
        check_tables(errors, manifest, document)

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
    workspace = root_manifest.get("workspace", {})
    package = workspace.get("package", {}) if isinstance(workspace, dict) else {}
    expected_workspace_metadata = {
        "version": VERSION,
        "edition": "2021",
        "rust-version": "1.85",
        "license": "MIT OR Apache-2.0",
        "readme": "README.md",
        "repository": CANONICAL_REPOSITORY,
    }
    if not isinstance(package, dict):
        errors.append("Cargo.toml:workspace.package: invalid table")
        package = {}
    for field, expected in expected_workspace_metadata.items():
        if package.get(field) != expected:
            errors.append(
                f"Cargo.toml:workspace.package.{field}: expected {expected!r}, "
                f"got {package.get(field)!r}"
            )
    workspace_dependencies = (
        workspace.get("dependencies", {}) if isinstance(workspace, dict) else {}
    )
    if not isinstance(workspace_dependencies, dict):
        errors.append("Cargo.toml:workspace.dependencies: invalid table")
    else:
        for name, value in workspace_dependencies.items():
            check_dependency(
                errors,
                ROOT / "Cargo.toml",
                "workspace.dependencies",
                name,
                value,
            )

    observed_published: set[str] = set()
    inherited = (
        "version",
        "edition",
        "rust-version",
        "license",
        "authors",
        "readme",
        "keywords",
        "repository",
    )
    workspace_members = workspace.get("members", []) if isinstance(workspace, dict) else []
    expected_members = [str(PACKAGE_PATHS[package]) for package in PACKAGES]
    if workspace_members != expected_members:
        errors.append(
            f"Cargo.toml:workspace.members: expected {expected_members!r}, "
            f"got {workspace_members!r}"
        )

    for manifest in manifests()[1:]:
        document = tomllib.loads(manifest.read_text())
        package_table = document.get("package")
        if not isinstance(package_table, dict) or package_table.get("publish") is False:
            continue
        name = package_table.get("name")
        if not isinstance(name, str):
            errors.append(f"{manifest.relative_to(ROOT)}: package name is missing")
            continue
        observed_published.add(name)
        for field in inherited:
            if package_table.get(field) != {"workspace": True}:
                errors.append(
                    f"{manifest.relative_to(ROOT)}:package.{field}: "
                    "must inherit the workspace release metadata"
                )
        description = package_table.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{manifest.relative_to(ROOT)}: package description is missing")
        expected_docs = f"https://docs.rs/{name}"
        if package_table.get("documentation") != expected_docs:
            errors.append(
                f"{manifest.relative_to(ROOT)}: documentation must be {expected_docs!r}"
            )
        observed_internal = observed_internal_dependencies(document)
        expected_internal = EXPECTED_INTERNAL_DEPENDENCIES[name]
        if observed_internal != expected_internal:
            errors.append(
                f"{manifest.relative_to(ROOT)}: internal dependency set is "
                f"{sorted(observed_internal)!r}, expected {sorted(expected_internal)!r}"
            )
    if observed_published != set(PACKAGES):
        errors.append(
            "publishable package set mismatch: "
            f"expected {sorted(PACKAGES)!r}, got {sorted(observed_published)!r}"
        )

    for notice in ("LICENSE-APACHE", "LICENSE-MIT", "THIRD_PARTY_NOTICES.md"):
        path = ROOT / notice
        if not path.is_file() or not path.read_bytes().strip():
            errors.append(f"required release notice is absent or empty: {notice}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        "manifest policy ok: exact dependencies, canonical repository metadata, "
        "three alpha.2 packages, canonical workspace paths, and complete root notices"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
