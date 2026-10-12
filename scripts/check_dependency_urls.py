#!/usr/bin/env python3
"""Fail closed on private dependency transports and optionally prove access."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import re
import subprocess
import tempfile
import tomllib

from release_packages import PACKAGE_PATHS, PACKAGES


ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_TRANSPORTS = ("ssh" + "://", "git" + "@", "git+ssh" + "://")


def manifests() -> tuple[Path, ...]:
    return (ROOT / "Cargo.toml",) + tuple(
        ROOT / PACKAGE_PATHS[package] / "Cargo.toml" for package in PACKAGES
    )


def git_dependencies(value: object) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        url = value.get("git")
        if url is not None:
            revision = value.get("rev")
            if not isinstance(url, str) or not isinstance(revision, str):
                raise ValueError("Git dependencies require string git and rev fields")
            if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ValueError(f"Git dependency {url!r} is not pinned to one full revision")
            found.append((url, revision))
        for child in value.values():
            found.extend(git_dependencies(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(git_dependencies(child))
    return found


def validate_source_text() -> list[str]:
    errors: list[str] = []
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.split(b"\0")
    for encoded in tracked:
        if not encoded:
            continue
        relative = encoded.decode("utf-8")
        path = ROOT / relative
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for forbidden in FORBIDDEN_TRANSPORTS:
            if forbidden in source:
                errors.append(f"{relative}: private dependency transport {forbidden!r}")
    return errors


def validate_manifests() -> tuple[list[str], set[tuple[str, str]]]:
    errors: list[str] = []
    dependencies: set[tuple[str, str]] = set()
    for manifest in manifests():
        try:
            document = tomllib.loads(manifest.read_text(encoding="utf-8"))
            discovered = git_dependencies(document)
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            errors.append(f"{manifest.relative_to(ROOT)}: {error}")
            continue
        for url, revision in discovered:
            if not url.startswith("https://"):
                errors.append(
                    f"{manifest.relative_to(ROOT)}: Git dependency is not public HTTPS: {url}"
                )
            dependencies.add((url, revision))

    lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_TRANSPORTS:
        if forbidden in lock:
            errors.append(f"Cargo.lock: private dependency transport {forbidden!r}")
    for url, revision in dependencies:
        identity = f"git+{url}?rev={revision}#{revision}"
        if identity not in lock:
            errors.append(f"Cargo.lock: missing exact source identity {identity}")
    return errors, dependencies


def prove_anonymous_access(dependencies: set[tuple[str, str]]) -> list[str]:
    errors: list[str] = []
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for url, revision in sorted(dependencies):
        with tempfile.TemporaryDirectory(prefix="kvpack-dependency-access.") as directory:
            subprocess.run(
                ["git", "init", "--bare", "--quiet", directory],
                check=True,
                capture_output=True,
                env=environment,
            )
            fetched = subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "-c",
                    "credential.helper=",
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    url,
                    revision,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            if fetched.returncode != 0:
                detail = fetched.stderr.strip().splitlines()[-1:] or ["fetch failed"]
                errors.append(f"anonymous fetch failed for {url}@{revision}: {detail[0]}")
                continue
            resolved = subprocess.run(
                ["git", "-C", directory, "rev-parse", "FETCH_HEAD"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            ).stdout.strip()
            if resolved != revision:
                errors.append(
                    f"anonymous fetch resolved the wrong revision for {url}: {resolved}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-anonymous", action="store_true")
    args = parser.parse_args()
    errors = validate_source_text()
    manifest_errors, dependencies = validate_manifests()
    errors.extend(manifest_errors)
    if args.check_anonymous and not errors:
        errors.extend(prove_anonymous_access(dependencies))
    if errors:
        print("\n".join(errors), file=__import__("sys").stderr)
        return 1
    suffix = " with anonymous exact-revision fetch" if args.check_anonymous else ""
    print(f"dependency URL policy ok: {len(dependencies)} exact HTTPS source(s){suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
