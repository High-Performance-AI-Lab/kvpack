# DwarfStar integration guide

## Verify the target revision first

DwarfStar is deliberately model-specific and its checkpoint format is active
code. Before editing, record the target commit and inspect `AGENT.md`, `ds4.h`,
`ds4.c`, `ds4_kvstore.h`, `ds4_kvstore.c`, and `ds4_distributed.c`. The baseline
inspected for this guide was upstream commit
`80ebbc396aee40eedc1d829222f3362d10fa4c6c`.

The baseline exposes:

- `ds4_session_save_payload` / `ds4_session_load_payload` for the canonical DS4
  payload stream;
- `ds4_session_stage_payload` for a stable temporary payload file;
- `ds4_session_save_snapshot` / `ds4_session_load_snapshot` for local in-memory
  snapshots;
- `ds4_session_save_layer_payload` / `ds4_session_load_layer_payload` for one
  distributed layer range;
- a disk cache in `ds4_kvstore.*` keyed first by rendered byte prefixes, with
  exact checkpoint tokens retained inside the payload.

The in-memory snapshot helper refuses distributed sessions at this revision.
The streaming payload path delegates distributed save/load to the coordinator,
which gathers and redistributes worker-owned layer payloads.

## Do not wrap the disk cache blindly

The `DSV4` payload is a composite native checkpoint. It includes header fields,
exact token IDs, next-token logits, per-layer row counts, logical raw KV rows,
compressed attention rows and frontiers, ratio-4 indexer rows and frontiers.
MTP draft state is invalidated and rebuilt after load.

That format is an excellent native round-trip oracle, but it is not one
fixed-width token-indexed state plane. The outer disk-cache key is a rendered
text SHA-1 and its compatibility header is intentionally narrower than
kvpack's semantic and representation identities. Do not import its filename as
a kvpack prefix identity, and do not store the entire payload through
`ExportSession` under a fake per-token shape.

Choose one implementation path explicitly:

1. **Native opaque artifact**: add a specified, bounded exact-cut blob surface
   to kvpack and use the versioned `DSV4` payload as its engine ABI. This is the
   shortest same-engine path but needs a real kvpack schema/API addition.
2. **Structured native representation**: expose logical DS4 state classes and
   map them to a richer versioned kvpack schema. Raw trailing rows can use a
   tail-window plane; compressed rows, variable row counts, and recurrent
   frontiers must be represented honestly rather than padded into ordinary KV.
3. **Portable representation**: define a new canonical layout shared with a
   second engine and add explicit transforms. Treat this as a separate research
   and qualification project.

Start with path 1 or 2 on one DwarfStar backend/build. Keep the existing DS4
serializer as the behavioral oracle throughout.

## State inventory

At each valid checkpoint include:

- authoritative `ds4_session_tokens` token IDs and checkpoint length;
- next-token logits if immediate sampling without another decode is part of
  resume semantics;
- live raw sliding-window rows in logical position order;
- compressed attention rows plus row counts and compressor frontier tensors;
- ratio-4 indexer rows plus row counts and indexer frontier tensors;
- context/raw-window/compressed-capacity and layout version;
- distributed layer ownership when saving shards;
- any future speculative state that becomes required for exact continuation.

Use DwarfStar's safe session boundary. UI prefill progress is explicitly not a
durable checkpoint boundary. Save only while the graph worker owns a complete,
valid token prefix and after accelerator synchronization.

## Identity requirements

Bind at minimum:

- the GGUF weights/config digest and DwarfStar model-shape identity;
- tokenizer, special tokens, and rendered chat protocol;
- `DS4_SESSION_PAYLOAD_VERSION` or the new structured ABI version;
- backend and qualified numerical lane;
- routed-expert quant profile, raw window, compression ratios, dtype/storage
  conversions, RoPE/position rules, and context compatibility;
- layer count/dimensions/vocabulary and MTP behavior;
- for distributed execution, exact layer ownership and route-compatible shard
  map.

Do not inherit the current disk cache's permissive cross-quant behavior unless
an explicit continuation qualification proves it and the qualified-math
identity records it.

## Transactional restore

The current `ds4_session_load_payload` validates substantial metadata but then
mutates the target session while reading. A production kvpack sink must not call
it on the live serving session.

Use this pattern:

1. `begin_restore` creates a fresh `ds4_session` with the authenticated context
   and route configuration, or allocates an inactive complete generation.
2. `write_verified_chunk` assembles the verified payload/planes only in that
   shadow session or staging file.
3. `commit_restore` invokes the DS4 load on the shadow, synchronizes all local
   and worker-owned state, verifies the resulting checkpoint length/tokens,
   then swaps the server's one mutable session under the graph-worker lock.
4. `abort_restore` destroys the shadow and any staged file.
5. `reset_restore` discards both sides of an uncertain swap and returns the
   worker to a known-empty checkpoint.

For distributed restore, the coordinator must not advertise success until all
layer shards are installed. A disconnect or shard failure aborts the whole
atomic group. Preserve DwarfStar's token-prefix hash/replay safeguards; kvpack
does not replace route recovery.

## Coexistence with `ds4_kvstore`

Avoid two independent eviction/catalog systems owning duplicate payloads.
During migration, select one authority:

- kvpack owns authenticated objects, exact-token lookup, retention, and
  publication; DwarfStar supplies serialization and installation; or
- the existing disk cache remains enabled as an explicit fallback while
  kvpack operates in a separate namespace and metrics identify which path hit.

Never treat a rendered-byte prefix match as sufficient authorization for a
kvpack restore. Re-derive and match the exact token IDs and full identities.

## DwarfStar qualification

- Compare uninterrupted execution, native DSV4 save/load, and kvpack restore
  at the same cuts.
- Compare next-token logits and deterministic continuations.
- Cover raw-window wrap, every compression ratio, indexer state, context growth
  allowed by the native loader, and incompatible smaller contexts.
- Test Metal, CUDA, ROCm, and CPU only as separately identified lanes; do not
  infer portability from a shared payload parser.
- Cover single-process and distributed payload paths, worker restart, missing
  layer ranges, and partial route failure.
- Corrupt headers, tokens, counts, each state class, and trailing bytes.
- Kill export/restore at each phase and prove the next request can recompute or
  restore cleanly.

Use the upstream source as the API authority:
<https://github.com/stefandsl/DwarfStar>.
