# kvpack architecture

Status: current implementation on `main`, crate version 0.1.0, 2026-07-27

This document describes the architecture that exists in this repository now.
It is not the Iodyne deployment proposal, a future distributed-cache design,
or a claim of universal cross-engine compatibility. Normative bytes are defined
by the specifications in [`spec/`](../spec/); this document explains how the
shipping components produce, validate, publish, find, and restore those bytes.

## 1. System definition

`kvpack` is an embedded, exact-state replay store for LLM inference engines.
An engine adapter supplies exact token IDs, compatibility identity, a complete
ordered state inventory, and the state bytes. `kvpack` turns those inputs into
an immutable pack, publishes it crash-safely, validates it before reuse, and
copies verified bytes into engine-owned buffers. The engine remains responsible
for the meaning of those bytes and for proving restored inference behavior.

The current system is local-file and single-host coordinated:

```text
inference engine
      │
      │ CacheEngineBackend
      ▼
ExactCacheCoordinator
      │
      ├── export ──► PackExportSink ──► PackSink ──► immutable .kvpack file
      │                                      │
      │                                      ├── optional pack-set pointer
      │                                      └── optional kvpackd publication
      │
      └── restore ◄─ PackRestoreSource ◄─ PackReader ◄─ sealed pack
                            │
                            └── exact identity + descriptor expectations

optional local services:
  StoreKey       NamespaceLease       WriteGovernor       kvpackd
  HMAC/KVENC     writer exclusion     endurance budget    memory admission
```

The large KV payload never passes through `kvpackd`. Readers open the pack file
and use positional I/O directly.

## 2. Ownership boundary

### kvpack owns

- deterministic pack-v1 framing and validation;
- exact-token and compatibility-namespace identifiers;
- bounded streaming file writes and bounded file validation/restores;
- commit inventory, Merkle verification, and whole-pack sealing;
- no-overwrite pack publication and atomic pack-set pointer replacement;
- adapter sequencing and mandatory abort/reset ordering;
- optional authenticated encryption, local admission, leases, write budgeting,
  and private telemetry primitives.

### The engine adapter owns

- synchronizing CPU/GPU/device work before export;
- enumerating every live state object and its true descriptor;
- exporting only initialized live bytes;
- preallocating restore destinations and installing bytes in native layout;
- choosing valid checkpoint boundaries and suffix-resume semantics;
- incrementing `engine_abi` whenever serialization or installation changes;
- proving logits/continuation equivalence after restore.

### The application/operator owns

- tokenization and checkpoint policy;
- store paths, permissions, retention, and garbage collection;
- holding `NamespaceLease` where required;
- calling the write governor and memory-admission APIs where required;
- serializing pack-set read/modify/publish operations;
- distributing and protecting store keys;
- external multi-host coordination, authentication, and authorization.

`kvpack` can prove that selected bytes match a committed artifact under an
exact identity. It cannot prove that an engine described its cache correctly.

## 3. Workspace and component graph

```text
                         ┌──────────────────┐
                         │   kvpack-core    │
                         │ pure wire codec  │
                         └────────┬─────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
          ┌────────────┐   ┌─────────────┐  ┌─────────────┐
          │   kvpack   │   │ conformance │  │ references  │
          │ file/store │   │ Rust tests  │  │ Python + C  │
          └──────┬─────┘   └─────────────┘  └─────────────┘
                 │
        ┌────────┼───────────┬──────────────┐
        ▼        ▼           ▼              ▼
   kvpack-ffi  kvpack-cli  kvpack-agent  kvpack-gateway
    C ABI       tools       AF_UNIX         TLS/h2
```

| Component | Shipping responsibility |
|---|---|
| `kvpack-core` | Pure in-memory pack codec, identifiers, record/commit/footer validation, Merkle tree, recovery, lossless and q8 codecs. It has no filesystem I/O and forbids unsafe Rust. |
| `kvpack` | File-backed sink/reader, validation pipeline, engine adapter contract, pack bridge, pack-set publication, store keys, encryption, namespace lease, admission ledger, write governor, and telemetry. |
| `kvpack-ffi` | ABI-versioned C writer/reader surface over `kvpack`; opaque handles, caller-owned buffers, stable error categories. Unsafe code is isolated here. |
| `kvpack-cli` | `validate`, `inspect`, deterministic `fixture`, `keygen`, pack-set `publish`, and KVENC `seal`/`open` commands. |
| `kvpack-agent` | Optional single-host AF_UNIX memory-admission coordinator and lifecycle service. |
| `kvpack-gateway` | Authenticated TLS 1.3/h2 gateway with bounded request and storage policy. |
| `conformance` | Pinned fixture/corruption corpus and cross-language verdict checks. |
| `reference/python` | Standard-library reference implementation used as an independent wire oracle. |
| `integrations` | Example llama.cpp and MLX adapters, not upstream engine patches. |

Most consumers need only the `kvpack` crate. It re-exports `kvpack-core` as
`kvpack::wire`.

## 4. Persistent data model

### 4.1 Pack structure

A sealed pack is one immutable aligned file:

```text
┌─────────────────────────────┐ offset 0
│ 4 KiB pack header           │ generation, namespace, model, writer, UUID
├─────────────────────────────┤ 4 KiB aligned
│ 512 B record header         │ type, descriptor, IDs, payload digest
│ payload                     │ exact or encoded state/metadata
│ zero padding                │ to next 4 KiB boundary
├─────────────────────────────┤
│ ... more records ...        │ strictly increasing sequence numbers
├─────────────────────────────┤
│ terminal COMMIT record      │ ordered inventory + Merkle root
├─────────────────────────────┤
│ 4 KiB footer                │ generation, commit offset, whole-pack SHA
└─────────────────────────────┘ EOF
```

The fixed sizes are:

| Structure | Size |
|---|---:|
| Pack header | 4,096 bytes |
| Record header | 512 bytes |
| Record alignment | 4,096 bytes |
| Commit fixed payload | 192 bytes |
| Commit entry | 72 bytes |
| Footer | 4,096 bytes |
| Maximum tensor rank | 8 |
| Maximum commit-chain depth | 16 |

Record types include exact token blocks, tensor deltas, terminal state,
logits, source/tool/index metadata, tombstones, and commit. State descriptors
bind cache kind, layer, dtype, shape, logical token span, live and allocated
bytes, and terminal/delta semantics. Unknown enums, invalid shapes, overflow,
reserved bits, and inconsistent semantics fail closed.

[`PACK_V1.md`](../spec/PACK_V1.md) is normative.

### 4.2 Integrity hierarchy

Integrity is layered rather than represented by one checksum:

1. The pack header and every record header have canonical SHA-256 digests.
2. Each record stores a SHA-256 payload digest.
3. Each object ID is content-derived from its sequence and payload.
4. The commit lists each prior committed record as sequence, object ID, and
   payload digest.
5. A canonical Merkle tree binds that ordered commit inventory.
6. The footer binds the header digest and SHA-256 of every byte through the
   terminal commit.

Validation also checks sequence/object uniqueness, parent links, exact zero
padding, terminal-commit placement, commit membership, required root/terminal
objects, parent-chain depth, footer offsets, and exact file length.

These hashes detect corruption and substitution inside the validated pack.
They are not authentication against an attacker able to rewrite an entire
unencrypted pack and all surrounding metadata. KVENC provides the optional
authenticated-encryption boundary described in section 12.

### 4.3 Exact identity

The engine-facing `CacheIdentity` has eight required fields:

1. model SHA-256;
2. model revision;
3. quantization;
4. adapter ID;
5. tokenizer SHA-256;
6. template SHA-256;
7. context-policy SHA-256;
8. engine ABI.

The Rust bridge hashes those fields into a model fingerprint and derives the
nine-field namespace HMAC required by pack-v1. Homogeneous descriptor sets use
their literal dtype/layout as namespace fields. Heterogeneous sets use ordered
schema digests that include layer, state name, and dtype/layout. The prefix ID
is a keyed HMAC over that namespace, the exact token count, and every token
encoded as little-endian `u32`.

The coordinator sorts state descriptors by `(layer, state_name)`. State names
are not written into pack record headers. The on-disk expectation check binds
the wire descriptor order; heterogeneous schema names are additionally bound
into the namespace. For a homogeneous schema, `engine_abi` and stable adapter
ordering are therefore load-bearing whenever two named states have otherwise
identical wire descriptors. Any identity, exact-token, or wire-descriptor
difference is a cache miss/rejection.

The store key makes namespace/prefix IDs private and prevents raw prompt text
from appearing in paths or telemetry. Exact token blocks remain inside the
pack because a restorer must prove the prefix bytes.

## 5. Write architecture

There are two writer implementations:

- `kvpack_core::PackWriter` is a one-shot in-memory encoder used by fixtures
  and small programmatic packs.
- `kvpack::PackSink` is the production file writer. `PackExportSink` connects
  it to an inference engine.

### 5.1 Engine export sequence

`ExactCacheCoordinator::export` enforces:

```text
engine.synchronize()
  -> inventory()
  -> sort and duplicate-check descriptors
  -> validate descriptor semantics
  -> sink.begin(identity, descriptors)
  -> engine.export_delta(descriptor, sink) for every descriptor
  -> sink.commit()
```

Any failure aborts the sink. The coordinator does not choose which states to
omit; the adapter's inventory must be complete for its declared ABI.

### 5.2 Pack bridge

`PackExportSink`:

- derives the model/namespace identity;
- writes the exact token block first;
- maps engine descriptors to pack record types and cache kinds;
- preserves the coordinator's sorted descriptor order;
- streams each state to `PackSink` in chunks no larger than 1 MiB;
- records the last terminal-state object where applicable;
- computes the exact prefix ID and emits one root commit.

This bridge currently emits exact/raw state records. Lossless and q8 codecs
exist at lower layers but are not selected automatically by the engine bridge.

### 5.3 File sink state machine

```text
create target
  -> create private 0600 sibling partial with create-new
  -> write 4 KiB pack header
  -> for each streamed state:
       write 512 B placeholder header
       write forward payload chunks (<= 1 MiB)
       hash payload and object ID incrementally
       verify expected byte count
       backpatch final 512 B header
       write zero padding to 4 KiB alignment
  -> append terminal commit
  -> sync_data
  -> reread committed bytes in bounded 4 MiB windows for whole-pack SHA
  -> append 4 KiB footer
  -> sync_all
  -> hard-link partial to final target without overwrite
  -> unlink partial
  -> fsync parent directory
```

The final target must not exist. The hard link makes publication atomic and
no-overwrite on the same filesystem. An aborted or dropped unfinished sink
syncs and preserves its private partial as evidence; it never creates the final
name.

`SinkOptions::uncached_io` maps to Darwin `F_NOCACHE` and is rejected on other
platforms.

## 6. Reader and validation architecture

### 6.1 Safe open and admission

`PackReader` opens with `O_RDONLY | O_CLOEXEC | O_NOFOLLOW`, then requires a
regular aligned file beneath the path chosen by the caller. Options can require
private file permissions. Before walking records, it validates file size,
header, footer, generation, header/footer linkage, committed-byte count, and
commit offset.

Default hard bounds include:

| Bound | Default/current value |
|---|---:|
| Maximum pack bytes | 8 GiB, caller configurable |
| Maximum retained token/commit record | 64 MiB |
| Sequential read chunk | 1 MiB |
| Pipeline threshold | 8 MiB |
| Pipeline I/O chunk | 4 MiB |
| Pipeline data windows | 8 (32 MiB total) |
| Default hash workers | min(4, available cores) |

The 8 GiB default is a safety configuration, not a pack-format limit.

### 6.2 Validation paths

Small packs use a sequential bounded walk. Large packs use a fan-out pipeline:

```text
one sequential pread thread (4 MiB chunks)
          │
          ├── framing consumer: headers, offsets, graph, commit
          ├── serial consumer: whole-pack SHA chain
          └── bounded hash workers: payload SHA, object ID, zero padding
```

The pipeline preserves the sequential reader's verdict and first-error
ordering; differential tests compare both paths. It never exposes state bytes
merely because a header parsed. A validation failure returns no trusted pack.

`CacheMode::ReadOnce` advises the OS not to retain large scans in page cache:
Darwin uses `F_NOCACHE`; Linux uses sequential advice and drop-behind. Cache
hints do not affect correctness.

### 6.3 Restore modes

`PackReader::open` validates the entire pack while retaining only bounded
metadata and record locations. `read_payload_into` then reads a selected state
record into an exact-sized caller buffer and re-verifies its payload digest.

That two-stage API reads selected payload bytes twice: once during validation
and once during delivery. `open_and_restore` instead captures every state
payload during the validation pass, subject to an explicit total restore-byte
cap, and returns nothing on any failure.

`read_payloads_parallel` may fill disjoint caller buffers from a shared file
descriptor. The first deterministic error cancels the operation and all
destinations are considered poisoned.

### 6.4 Expectation layer

Pack structural validity does not authorize a restore. `verify_expectations`
separately recomputes and compares:

- the keyed namespace;
- the exact token bytes and keyed prefix ID;
- the expected ordered state-record schema.

`PackRestoreSource` applies this expectation layer and maps descriptor position
to validated record position. This prevents a valid pack for another model,
token prefix, dtype/layout, or state inventory from being installed.

## 7. Atomic engine restore

Before an engine buffer is touched, `ExactCacheCoordinator::restore`:

1. synchronizes the engine;
2. obtains, sorts, and validates the current inventory;
3. compares the complete `CacheIdentity` and descriptor sequence with the
   validated restore artifact.

Only then does it enter the mutation boundary:

```text
engine.begin_restore()
  -> restore_preallocated(each descriptor)
  -> install_terminal_states(all terminal descriptors)
  -> engine.commit_restore()
```

On any error after `begin_restore`, it calls `abort_restore`, captures that
error if present, and then calls `reset_cache` unconditionally. A reset failure
outranks the original error because engine state is then unknown.

Atomicity here is an engine contract, not a filesystem trick. The adapter must
stage bytes or otherwise ensure that partially restored state cannot become
live before `commit_restore`.

## 8. Publication, discovery, and recovery

The current repository has two separate discovery mechanisms. They are not a
single distributed catalog.

### 8.1 Pack-set pointer library

`publish_pack_set`:

1. canonicalizes every pack path;
2. rebuilds and validates the complete prefix index in generation order;
3. writes a small private JSON pointer to a create-new temporary file;
4. fsyncs it;
5. atomically renames it over the pointer;
6. fsyncs the parent directory.

Readers parse the pointer strictly, rebuild an in-memory index, and call
`longest_prefix`, which tests keyed prefix IDs from the full token sequence
down to the empty prefix.

Current limitations are important:

- index rebuild reads each pack fully into memory through `std::fs::read`;
  unlike `PackReader`, this administration path is not a bounded large-pack
  scanner;
- concurrent pack-set updates are atomic but not merged; callers must
  serialize read/modify/publish or the last writer can replace another update;
- the pointer stores canonical filesystem paths and is local-filesystem state,
  not a remote catalog protocol.

### 8.2 kvpackd index

`kvpackd` has a separate in-memory `prefix_id -> (path, materialized bytes)`
map populated by a local `Publish` request. At publish time it checks only that
the path is a regular file; it does not reread or fully validate the pack. The
publisher must have sealed/validated it, and every restoring client still uses
`PackReader`.

The daemon index and pack reference counts are volatile and must be repopulated
after daemon restart. The memory-admission ledger is durable; the discovery
index is not. `kvpackd` does not consume the pack-set pointer automatically.

### 8.3 Core recovery API

`kvpack_core::recover_pack` scans an in-memory byte slice and returns the
verified committed prefix plus the first tail error. This supports inspection
and recovery tooling. Strict `validate_pack` rejects any recovery error,
missing commit, or required missing footer.

Published file-backed restore uses sealed packs. It never guesses or exposes an
incomplete tail.

## 9. Concurrency model

| Operation | Current behavior |
|---|---|
| Multiple readers of one sealed pack | Supported across threads/processes with positional reads. |
| Reader while a pack is being built | Safe: the final name is absent until seal and hard-link publication. |
| Multiple writers in different namespaces | Supported when they target different pack files. |
| Multiple writers to one pack | Unsupported; one `PackSink` owns one pack. |
| Race for one final path | First no-overwrite link wins; later writer fails. |
| Same-namespace writers on one host | Caller may hold `NamespaceLease` (`flock`) across write/publication. |
| Same-namespace writers across hosts | Not coordinated by shipping kvpack. |
| Concurrent pack-set publishers | Atomic replacement, but no merge/CAS; caller must serialize. |
| Reader during GC/unlink | An already-open Unix descriptor can finish; direct readers do not prevent unlink. |
| GC with kvpackd restore grant | Daemon pack ref-count blocks removal while the grant is active, if GC consults `pack_removable`. |

`NamespaceLease`, `WriteGovernor`, and `kvpackd` are opt-in. `PackSink` does not
acquire them automatically.

## 10. kvpackd single-host service

`kvpackd` is an optional AF_UNIX control service. It is deliberately not a
storage server.

```text
client ──64 KiB-capped control frames──► kvpackd
                                         │
                                         ├── durable admission ledger
                                         ├── volatile prefix index
                                         ├── volatile pack ref-counts
                                         └── condvar wait/reaper

client ◄──────── admitted pack path ─────┘
  │
  └── opens and validates pack directly
```

Its responsibilities are:

- reserve the materialized restore peak before a client allocates it;
- enforce total and tighter wired-memory caps;
- optionally wait for released budget;
- renew/release reservations;
- reclaim expired or dead-process reservations;
- ref-count packs associated with active restore grants;
- return a locally published pack path for an exact prefix ID.

The admission log is private, append-only, hash-chained, `flock`-serialized,
and fsynced. Reservations are PID-tagged and have a default 30-second TTL.
Connection close is not the release signal; explicit release, expiry, or
detected process death is.

Default policy derives an application cap of 65% of physical RAM, admission at
90% of that application cap, and wired memory at 60% of the application cap.
Operators can provide physical bytes explicitly.

Current trust and scope boundaries:

- AF_UNIX only; no TCP, remote transport, mTLS, or tenant authentication;
- the client declares its PID in `Hello`; this is a trusted local-client
  protocol, not peer-credential authorization;
- socket access is controlled only by the chosen path, parent-directory
  permissions, and process umask; the daemon does not authenticate peers, so
  the default `/tmp/kvpackd.sock` is a development convenience rather than a
  hardened deployment location;
- published path and materialized-byte count are caller supplied;
- the server caps frames before allocation and defaults to 512 connections;
- the daemon cannot reclaim memory already materialized inside a live client.

## 11. Operational controls

### Namespace lease

`NamespaceLease` uses an exclusive nonblocking `flock` on a private per-
namespace lease file. The kernel releases it after process death. It is valid
on one host only and is not acquired by the writer automatically.

### Memory admission

`AdmissionLedger` is available directly or through `kvpackd`. It reserves the
true materialized peak, not compressed pack bytes, records every transition
durably, reconciles dead PIDs on open, and fails closed above configured caps.

### Write-endurance governor

`WriteGovernor` maintains a private append-only 24-hour ledger and charges the
greater of caller-reported logical and attributed physical bytes. Its default
ceiling is 0.82 TB/day with lower thresholds for background and normal work.
It is single-host and advisory to the application: the sink does not call it.

### Telemetry

`append_telemetry` writes one canonical private JSON line with event, generation,
prefix ID, sequence, timestamp, and numeric value. It uses append, one write,
and fsync. It is separate from packs so cache hits never mutate pack bytes. The
API has no fields for prompts, raw tokens, or payloads.

## 12. Store keys and encryption

### Store key

A `StoreKey` is 32 bytes in a `0600` file under an explicitly supplied allowed
root. Creation uses a private temporary, fsync, no-overwrite hard link, and
directory fsync. Loading resolves paths, rejects symlink/file-type/permission
violations, and never exposes key bytes through `Debug` or CLI output; only a
SHA-256 fingerprint is shown.

The key drives namespace/prefix HMACs. The KVENC file helpers also accept a
`StoreKey` as their root key.

### Optional KVENC envelope

KVENC is outside pack-v1. It wraps a complete pack in a chunked
ChaCha20-Poly1305 envelope with HKDF-SHA-256-derived key material and a random
salt. Default plaintext chunks are 64 KiB. The envelope authenticates its
header and every chunk; reorder, truncation, wrong key, and byte tampering fail.

The current `seal_pack_file` and `open_pack_file` convenience APIs read the
complete input into memory and return/write a complete buffer. They are not a
bounded multi-gigabyte streaming encryption path. A caller must decrypt before
passing plaintext pack bytes to `PackReader`.

[`KVENC_V1.md`](../spec/KVENC_V1.md) is normative.

## 13. Codec lanes

| Lane | Current status |
|---|---|
| Exact/raw | Shipping correctness lane; payload bytes restore unchanged. |
| Lossless | Shipping exact lane using the versioned zlib envelope and bounded codec APIs. |
| Q8 | Deterministic implementation and model-free tests exist; research-only, not release-qualified for inference state. |
| Q4/Q2 | Wire enum values are reserved but no v1 semantic encoding exists. |

Codec validation is below the pack layer: the outer record digest covers stored
bytes, while codec envelopes independently bind source/restored sizes and
digests. The engine bridge does not automatically select a codec.

## 14. Public integration surfaces

### Rust

The primary integration implements `CacheEngineBackend` and uses:

- `ExactCacheCoordinator` for export/restore ordering;
- `PackExportSink` for exact pack creation;
- `PackRestoreSource` for expectation-checked restore;
- lower-level `PackSink`/`PackReader` when the application owns more policy.

### C ABI

`include/kvpack.h` exposes ABI version 1:

- create/begin/write/commit/abort a streaming pack sink;
- validate/open a pack reader;
- inspect record metadata and shapes;
- copy one payload or parallel payloads into caller buffers;
- retrieve stable status categories and a thread-local error message.

The FFI never allocates caller output buffers. Buffers are poisoned after an
error. Reader operations may be concurrent; handle close must be exclusive.

### CLI

The `kvpack` binary is an administration/debugging surface, not a daemon:

- `validate` and `inspect` open a sealed pack through `PackReader`;
- `fixture` writes the deterministic conformance pack;
- `keygen` creates a rooted store key;
- `publish` validates a pack set and atomically replaces its pointer;
- `seal`/`open` wrap or unwrap a full file with KVENC.

### Reference integrations

- llama.cpp v0 stores one opaque sequence-state blob and verifies a bitwise
  round trip. It does not yet expose structured per-layer cross-engine state.
- MLX v0 maps plain attention `KVCache` arrays through the C ABI and verifies
  f16/f32/bf16 round trips. Rotating and quantized MLX caches are not covered.

These integrations demonstrate the boundary; they are not bundled changes to
upstream inference engines.

## 15. Failure semantics

The design consistently turns uncertainty into a miss or explicit error:

| Failure | Result |
|---|---|
| Writer crashes before publication | Private partial may remain; final target is absent. |
| Final target appears concurrently | Writer refuses overwrite. |
| Record/commit/footer corruption | Validation rejects the pack. |
| Truncated or invalid tail | Recovery may report the prior verified commit; sealed restore rejects the file. |
| Wrong tokens/identity/schema | Expectation validation rejects before engine mutation. |
| Payload read fails during restore | Destination is poisoned; coordinator aborts and resets engine cache. |
| Memory budget unavailable | `kvpackd` returns busy/exceeds-cap; caller recomputes or retries. |
| Daemon/index unavailable | Direct local APIs remain usable; application treats lookup as a miss. |
| Lease unavailable | Application waits, routes elsewhere, or skips the write. |
| Write budget exhausted | Optional governor rejects according to purpose tier. |

No repair path fabricates missing records, guesses state shape, substitutes a
nearby prefix, or accepts semantic similarity.

## 16. Shipping deployment shapes

### Embedded library only

One process holds the key, writes packs with `PackExportSink`, keeps its own
index or pack-set pointer, and restores with `PackRestoreSource`. This is the
smallest and most complete path.

### Multiple local processes

Processes share pack files, an optional `NamespaceLease`, pack-set pointer, and
optional write/admission ledgers. Readers independently validate packs. The
application serializes pointer updates and GC.

### Local daemon-assisted serving

Applications publish prefix/path/peak-memory metadata to `kvpackd`, request an
admitted restore path, open and validate it directly, then release/renew the
grant. The daemon must be repopulated after restart and remains a trusted
same-host component.

### Portable file exchange

A sealed pack can be copied to another compatible machine and validated there.
That portability is a file property, not a shipping network service. The
receiving engine must satisfy the same exact identity/schema contract and prove
its own numerical compatibility.

## 17. Explicit non-features and current limitations

The current shipping architecture does **not** provide:

- a remote or distributed cache service;
- TCP/RDMA/Thunderbolt transport;
- multi-host leases, generation CAS, or a durable distributed catalog;
- authentication/authorization for remote clients or tenants;
- universal CUDA/Metal/native-layout conversion;
- automatic engine discovery, tokenization, checkpoint selection, or routing;
- semantic/approximate KV lookup;
- proof of numerical equivalence on behalf of an engine;
- bounded streaming for the pack-set index rebuild or KVENC file helpers;
- automatic retention/compaction/GC orchestration;
- automatic enforcement of namespace leases or write-endurance policy.

The Iodyne/DGX/Mac design is a proposed deployment around these current pack
and restore primitives. Its missing distributed services are not part of
shipping `kvpack`; see [the Iodyne design](IODYNE_DGX_MAC_IO_SPEC.md).

## 18. Source map

| Concern | Primary implementation/specification |
|---|---|
| Pack bytes and invariants | [`PACK_V1.md`](../spec/PACK_V1.md), [`kvpack-core`](../crates/kvpack-core) |
| Streaming writer | [`sink.rs`](../crates/kvpack/src/sink.rs) |
| Bounded reader/restore | [`reader`](../crates/kvpack/src/reader), [`pipeline`](../crates/kvpack/src/pipeline) |
| Engine contract | [`adapter.rs`](../crates/kvpack/src/adapter.rs) |
| Pack/engine bridge | [`bridge.rs`](../crates/kvpack/src/bridge.rs) |
| Pack-set publication/index | [`publish.rs`](../crates/kvpack/src/publish.rs) |
| Single-host writer lease | [`lease.rs`](../crates/kvpack/src/lease.rs) |
| Admission | [`admission`](../crates/kvpack/src/admission), [`kvpackd`](../crates/kvpackd) |
| Write endurance | [`governor.rs`](../crates/kvpack/src/governor.rs) |
| Store key and KVENC | [`store_key.rs`](../crates/kvpack/src/store_key.rs), [`encrypt.rs`](../crates/kvpack/src/encrypt.rs), [`KVENC_V1.md`](../spec/KVENC_V1.md) |
| C ABI | [`kvpack-ffi`](../crates/kvpack-ffi), [`kvpack.h`](../include/kvpack.h) |
| Conformance | [`TESTING.md`](TESTING.md), [`conformance`](../conformance), [`reference/python`](../reference/python) |

The architecture changes when these ownership or data-flow boundaries change,
not merely when a new adapter or benchmark is added.
