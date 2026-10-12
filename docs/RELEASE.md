# Build, platform, and release contract

This document describes the draft package contract. It does not freeze the
durable cache bytes or authorize publishing, tagging, signing, merging to
`main`, or deployment. Private candidate and evidence feature-branch pushes are
permitted; release publication actions remain outside this worktree's authority.

## Toolchains and language boundaries

The workspace MSRV is Rust 1.85 with edition 2021. The local release battery
tests the MSRV and current stable compiler as available. Automatic GitHub
Actions triggers are disabled; the tracked workflow is manual-only. Raising
the MSRV is a public compatibility change and requires an explicit release
decision.

## Platform matrix

| Platform | Draft build/test lane | Service boundary | Release status |
| --- | --- | --- | --- |
| macOS on Apple Silicon | All three Rust crates and handoff receiver | Unix store primitives and rustls receiver | Primary development lane |
| Linux x86-64/aarch64 | All three Rust crates and handoff receiver | Unix store primitives and rustls receiver | Manual qualification required while hosted CI is disabled |
| Windows | Core codec may compile, but the store contract is Unix-specific | None | Unsupported for the alpha.2 release |

RDMA, Thunderbolt RDMA, GPUDirect, and direct peer transfer are not inferred
from an operating system or architecture. They remain unavailable unless a
separately versioned transport profile is independently probed and qualified.

## Package contents and licenses

The three alpha.2 release crates are `kvpack-core`, `kvpack`, and
`kvpack-handoff`. Each manifest carries an exact
draft version, MSRV, dual `MIT OR Apache-2.0` license expression, author,
README, documentation URL, description, and exact registry dependencies. The
repository URL is the canonical public project repository,
`https://github.com/High-Performance-AI-Lab/kvpack`. The package policy rejects workspace or
package-set drift.

Project license texts are in [`LICENSE-MIT`](../LICENSE-MIT) and
[`LICENSE-APACHE`](../LICENSE-APACHE). The checksum-pinned dependency inventory
is [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Every draft archive
contains the complete three files, and the archive policy verifies their exact
bytes rather than accepting manifest metadata alone. Regenerate or verify the
notice inventory offline with:

```sh
python3 scripts/generate_third_party_notices.py --check
python3 scripts/check_dependency_urls.py
```

The notice check reads every registry package in `Cargo.lock`, verifies the
cached `.crate` archive against the locked SHA-256, requires declared license
metadata (or an included license file), and rejects every non-crates.io
dependency source. `scripts/check_manifest_policy.py` allows only canonical
same-workspace paths paired with exact alpha.2 requirements; the archive check
then proves Cargo removed those paths from normalized package manifests. A current
advisory-database scan is captured in the V1 evidence campaign because advisory
state changes independently of source bytes.

The dependency URL policy rejects private Git transports. If a future current
workspace manifest adds a Git dependency, it must use public HTTPS, a full
40-character revision, and the exact lockfile identity.

## Offline package qualification

Fetch the exact lock once, then perform the qualification without network
access:

```sh
cargo fetch --locked
scripts/package_draft.sh
```

The script verifies all cached dependency archive checksums, builds a disposable
local Cargo registry, packages the three workspace crates, and indexes the
resulting archives. Nothing is published. A consumer then installs only from that registry and
executes the real safe Rust path: store creation, one-pass export, exact lookup,
authenticated plan construction, scatter preparation, detached restore, atomic
commit, and engine-free acknowledgement, and checks the handoff package identity.
Finally, each unpacked archive builds all of its targets offline.

To retain draft archives and their aggregate digest manifest outside the
worktree, set a new or empty output directory:

```sh
KVPACK_PACKAGE_OUTPUT_DIR=/absolute/evidence/packages \
  scripts/package_draft.sh
```

`scripts/check_reproducible_packages.sh` packages the same source twice in
independent temporary registries and requires byte-identical `.crate` SHA-256
manifests. Raw compiler logs belong in ignored/external evidence storage; the
tracked evidence bundle records command, aggregate verdict, and log digest.

## Candidate release sequence

1. Start from one clean local candidate with exact `Cargo.lock` and no source
   changes during qualification.
2. Run formatting, clippy with warnings denied, rustdoc with warnings denied,
   locked workspace tests, deterministic wire vectors, LOC, manifest and
   dependency policy, dependency notices, offline package smoke, and package
   reproducibility.
3. Run the V1 resource, restart, scale, target-storage, TCP, long-soak, and
   fault campaigns against that same candidate and record real blockers.
4. Only after all R0–V1 rows are green, perform Z1's field/byte review and rerun
   the complete campaign without format changes.
5. Create the local format-freeze commit. Merging to `main`, tagging, signing,
   publishing, and deployment remain pending separate release authority.

Ferrite is not part of this package gate. Any later consumer qualification must
use a dedicated kvpack integration worktree after the format freeze; Ferrite
`main` remains untouched.
