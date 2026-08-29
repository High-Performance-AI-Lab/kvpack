# kvpack

**A fast, safe KV-cache replay layer for LLM inference.**

kvpack lets an inference engine save already-computed KV-cache and recurrent
state, then restore it after a process restart, a crash, or on another
compatible machine. On a cache hit, the engine can resume from the restored
state instead of prefilling the same tokens again.

Typical uses include long system prompts, repeated RAG documents, recurring
sessions, and precomputed contexts. Replay is useful whenever loading the saved
state is cheaper than recomputing it.

Here, **replay means restoring inference state**. kvpack is not an inference
engine, a semantic prompt cache, or a request/response log. It stores and
verifies the exact bytes an engine adapter gives it; the engine remains
responsible for what those bytes mean.

See [the current shipping architecture](docs/ARCHITECTURE.md) for the crate
graph, write/read paths, publication and recovery model, daemon boundary,
concurrency, security, and explicit non-features.

## How it works

The lifecycle has two fail-closed paths: durable publication on the first
execution, then authenticated exact-prefix replay on a later execution.

```mermaid
sequenceDiagram
    accTitle: kvpack publication and replay lifecycle
    accDescr: An inference engine publishes authenticated state, then restores the longest exact compatible prefix into shadow state before atomic installation.
    autonumber
    participant E as Inference engine + adapter
    participant K as kvpack
    participant S as Immutable objects + catalog

    Note over E,S: First execution: publish a reusable prefix
    E->>K: Exact token IDs, compatibility identity, and complete state inventory
    K->>S: Authenticate and stage immutable chunks
    K->>S: fsync, then publish without overwrite
    K->>S: Commit manifest and exact-prefix index atomically

    Note over E,S: Later execution: restore or recompute
    E->>K: Request token IDs and compatibility identity
    K->>S: Find the longest exact-prefix candidate
    S-->>K: Manifest chain and immutable chunks
    K->>K: Verify identity, chain, chunks, and resource limits
    alt Valid exact-prefix candidate
        K->>E: Verify and scatter bytes into engine-invisible shadow state
        alt Complete restore succeeds
            E->>E: Install atomically and compute only the uncached suffix
        else Restore or installation fails
            K-->>E: Abort or reset so no partial state remains live
            E->>E: Prefill normally
        end
    else No valid exact prefix
        K-->>E: Cache miss without mutating live state
        E->>E: Prefill normally
    end
```

1. An engine adapter describes its live state and supplies the exact token IDs
   that produced it.
2. kvpack binds that state to the model, revision, quantization, adapter,
   tokenizer, chat template, context policy, engine ABI, dtype, and layout.
3. State is streamed into a pack and published only after its terminal commit
   and integrity seal are durable.
4. For a later request, kvpack can find the longest stored exact-token prefix,
   validate the pack and its compatibility metadata, and stream the payloads
   into engine-owned buffers.
5. The engine installs the restored state and computes only the remaining
   suffix. If restore fails after it starts, the coordinator aborts and resets
   the engine cache rather than leaving partially installed state live.

Exact-request replay is the simplest integration: save a state for a token
sequence and restore it when the same sequence appears again. Partial-prefix
replay uses the same store and longest-prefix lookup, but the adapter must know
how to install a checkpoint at token *N* and continue from *N*. kvpack does not
infer that policy from opaque engine bytes.

## What kvpack provides

- **Exact state replay** — raw and losslessly encoded state restores to the
  original bytes. Ordinary KV, MLA, sliding/rotating KV, and recurrent-state
  families have explicit storage semantics.
- **Fail-closed compatibility** — keyed cache identities include every runtime
  input that can change state meaning. A different model, tokenizer, template,
  quantization, layout, or engine ABI is a miss, not a best-effort restore.
- **Crash-safe publication** — packs are append-only, commits are written last,
  and pack sets are published atomically. A torn write cannot replace the last
  known-good generation.
- **Layered integrity checks** — record headers and payloads are hashed,
  commits are Merkle-verified, and sealed packs carry a whole-pack digest.
  Corruption, truncation, and single-bit changes are rejected.
- **Content-addressed prefix lookup** — cache keys bind an exact token sequence
  to its compatibility namespace without putting raw prompt text in paths or
  telemetry. The index supports longest committed-prefix lookup and tombstones.
- **Bounded, parallel restore** — the reader uses bounded positional reads and
  copies state directly into caller-provided buffers while re-verifying each
  payload.
- **Optional encryption at rest** — the `kvenc` envelope uses
  ChaCha20-Poly1305 with HKDF-derived keys. Plain packs retain corruption
  detection; encrypted envelopes additionally provide authenticated encryption.
- **Single-host operational controls** — per-namespace writer leases, memory
  admission, a rolling write-endurance governor, and an optional coordinator
  daemon are included.
- **Rust storage and handoff surfaces** — use the native storage adapter contract
  or the authenticated live-handoff framing crate.

## The integration boundary

kvpack owns:

- the pack format, durable write protocol, validation, indexing, and restore
  I/O;
- exact-token and compatibility-identity matching;
- failure ordering around export and restore;
- byte-for-byte preservation of the state supplied by the adapter.

The engine adapter owns:

- synchronizing device work before export;
- describing and serializing every live state object;
- allocating, installing, and committing restored state;
- deciding where prefix checkpoints are valid and how suffix execution resumes;
- changing `engine_abi` whenever the serialized layout or restore semantics
  change;
- proving that restore produces the same model behavior as uninterrupted
  execution.

That division is deliberate: kvpack can prove that the saved bytes came back
unchanged and were selected under the expected identity. Only the engine can
prove that those bytes are a correct KV cache for its runtime.

## Concurrency model

kvpack is multi-reader and supports coordinated multi-writer stores, but every
individual pack is single-writer and immutable.

| Scenario | Support |
|---|---|
| Multiple readers of one sealed pack | Yes, across threads and processes. |
| A reader arriving while a pack is being written | Safe: the target is absent until the fully sealed private partial is published atomically. |
| Multiple writers contributing to one pack | No. One `PackSink` owns one pack generation and writes its single terminal commit. |
| Multiple writers racing for the same target path | Fail-closed: no overwrite occurs; the first publication wins and later publishers receive an error. |
| Multiple writers producing separate packs | Yes, subject to the namespace and catalog rules below. |
| Writers in different namespaces | Yes. Independent namespaces can be written concurrently. |
| Writers in the same namespace | Serialize them with `NamespaceLease` on one host. |
| Multiple readers during pack retirement or GC | Use `kvpackd` restore grants when GC must respect active readers; it reference-counts granted packs. |
| Concurrent replacement of one pack-set pointer | Publication is atomic, but updates are not merged. Callers must serialize read-modify-publish operations to avoid a last-writer-wins lost update. |

Readers use positional reads and never mutate a sealed pack. A `PackReader` can
serve parallel restores over one shared file descriptor, and separate processes
can open the same pack concurrently. On Unix, a reader that already holds an
open descriptor can finish after the path is unlinked, but direct readers do not
prevent that unlink; use the daemon's restore admission and pack references when
coordinating garbage collection.

Writer coordination is explicit. `NamespaceLease` is a single-host advisory
lock: acquiring it excludes another writer for the same namespace while leaving
other namespaces independent. `PackSink` and `PackExportSink` do not acquire the
lease automatically, so the application must hold it across the write and any
related catalog publication. The lock does not coordinate different machines;
partition namespaces by host or provide external cross-host coordination.

## Integrating an engine

### Rust

Implement [`CacheEngineBackend`](crates/kvpack/src/adapter.rs). The
`ExactCacheCoordinator` drives synchronized export, checks identity and the
complete descriptor sequence before touching engine buffers, and enforces
abort-plus-reset on restore failure.

[`PackExportSink` and `PackRestoreSource`](crates/kvpack/src/bridge.rs) connect
that contract to pack files. The lower-level `PackSink` and `PackReader` APIs
are also available when an engine needs to own more of the lifecycle.

### Live handoff

`kvpack-handoff` provides fail-closed framing, manifests, incremental
verification, sealed bundles, and the optional authenticated receiver used by
the disaggregated prefill lane. Peer authentication and engine-specific cache
installation remain caller responsibilities.

The alpha.2 release workspace is Rust-only. Legacy C ABI, CLI, agent, gateway,
conformance, and benchmark source directories remain in the repository for
historical development but are not workspace members or alpha.2 packages.

## What kvpack does not do

- It does not tokenize prompts, choose cache checkpoints, or perform semantic
  similarity matching. Prefix identity is over exact token IDs.
- It does not define a universal in-memory KV layout. Cross-engine reuse is
  possible only when adapters deliberately share a compatible representation.
- It does not run inference or guarantee numerical equivalence on behalf of an
  engine. Adapter-level restore-and-continue tests remain required.
- It does not provide a distributed cache service. The handoff crate defines
  authenticated framing, not discovery, scheduling, or a multi-tenant service.
- Lossy q8 state storage is research-only and is not release-qualified. Exact
  raw and lossless replay are the supported correctness lanes.

## Pack v1 safety model

The normative specifications live in [`spec/`](spec/):
[`PACK_V1.md`](spec/PACK_V1.md), [`KVENC_V1.md`](spec/KVENC_V1.md),
[`LOSSLESS_V1.md`](spec/LOSSLESS_V1.md), and [`Q8_V1.md`](spec/Q8_V1.md).
The encoding is deterministic, so independent implementations can produce and
validate identical bytes.

An unencrypted pack's hashes detect accidental corruption; they are not a
substitute for authentication against an attacker who can rewrite the entire
pack. Use a `kvenc` envelope when authenticated encryption at rest is required,
and protect the store key as a secret.

## Conformance and testing

```sh
cargo test --workspace --all-targets --locked
scripts/check_wire_vectors.sh
```

The deterministic wire-vector digest manifest is SHA-256-pinned. The Rust test
suite covers truncation and bit-flip rejection, randomized store operations,
crash consistency, restore equivalence, and handoff validation. The complete
manual candidate gates are recorded in [`docs/RELEASE.md`](docs/RELEASE.md).

## Repository layout

- `crates/kvpack-core` — pure in-memory pack-v1 codec with no file I/O.
- `crates/kvpack` — store, adapter contract, pack bridge, reader/writer,
  publication, admission, encryption, and operational controls.
- `crates/kvpack-handoff` — authenticated live-KV framing, verification, and
  sealed bundle handling.
- `reference/python/` — stdlib-only Python reference implementation.
- `spec/` — normative wire specifications.

Only the three crates above are members of the alpha.2 workspace.

## Requirements

- Rust 1.85 or newer.
- A Unix-like host for the store layer (`flock`, `pread`/`pwrite`, hard links).
- `python3` for release-policy and deterministic-vector checks.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.
