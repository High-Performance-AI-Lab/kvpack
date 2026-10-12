# Canonical family, realized schema, and cut manifest

> **Pre-freeze draft.** No compatibility promise applies before Z1.

All integers and state keys follow `IDENTITY_V1.md`. Counts precede their exact
number of entries. No optional section, extension block, sentinel tail, or
trailing byte is accepted.

## Representation family (`KVFAM1\0\0`)

The standalone object is:

```text
magic[8] || u16 version=1 || u16 zero || family_body
```

The family body is:

```text
engine_cache_abi Id32
representation_mode u16       # native=1, portable=2
zero u16
page_size_tokens u32          # nonzero
topology Id32
shard_map Id32
state_count u32
state[state_count]
```

Each static state is:

```text
state_key
cache_kind u16                # ordinary KV=1 only
dtype u16                     # fixed-width DType value
codec u16                     # raw=1, lossless=2 only
codec_version u16             # exactly 1
layout u16                    # contiguous=1, strided=2
token_axis_rule u16           # direct=1, gather=2, TailWindow=3
token_axis u8
zero u8
elements_per_token u64
rank u8                       # 1..=8
dimension[rank] u64           # zero means token; fixed dimensions are nonzero
dependency_count u16
dependency_state_key[dependency_count]
```

There is exactly one zero dimension and its index equals `token_axis`.
`elements_per_token` equals the checked product of every fixed dimension.
States and dependency lists are unique canonical order and dependencies form a
closed acyclic graph. Portable mode requires contiguous states using `direct`
or `TailWindow`; `gather` is not portable.

## Realized cut schema (`KVRCS1\0\0`)

The standalone object is `magic || u16 version=1 || u16 zero || schema_body`.
The schema body begins with a manifest-kind union:

```text
kind u8 || depth u8 || zero u16
```

`kind=0` is full and requires `depth=0` with no union payload. `kind=1` is delta
and is followed by `parent_manifest Id32 || parent InputCutId`; depth is 1..=7.
All other tags are unknown.

Next are:

```text
state_count u32
realized_state[state_count]
atomic_group_count u32
atomic_group[atomic_group_count]
segment_restored_bytes u64
complete_restored_bytes u64
```

Each realized state is:

```text
state_key
full_shape                    # u8 rank, then rank*u64 dimensions
segment_shape
stride_count u8
stride[stride_count] u64      # element strides
logical_start u64
logical_count u64
physical_offset_bytes u64
physical_span_bytes u64
complete_physical_bytes u64
absolute_position u64
window u64                    # exactly zero for ordinary causal KV
chunk_span_count u32
chunk_span[chunk_span_count]
```

A chunk span is `u64 token_start || u64 token_count || u64 plaintext_offset ||
u32 plaintext_bytes`. `plaintext_offset` is absolute from the beginning of the
complete logical state stream, not relative to a delta segment. Spans partition
the state's covered token and absolute byte ranges without gaps, overlap, empty
entries, or token splitting. `plaintext_bytes` equals token count times the
family's exact bytes per token and is at most 4 MiB. Absolute offsets let the
same authenticated chunk object appear unchanged in a delta and a later
reference-only full compaction.

An atomic group is `u32 nonzero_id || u32 state_count || state_key[...]`.
Groups sort by ID, member keys sort canonically, and together they partition the
complete family inventory exactly once.

A `direct` full state covers `[0, child_cut)`. A `direct` delta state covers
exactly `[parent_cut, child_cut)`. A `TailWindow` state is legal only in a full
manifest and covers one nonempty trailing range `[child_cut-count, child_cut)`;
its full and segment shapes both substitute that stored count at the token
axis. Its physical offset is zero because the artifact stores a compact linear
tail, while chunk `plaintext_offset` remains absolute in the logical stream so
an engine can place the verified bytes at the authenticated token positions.
`TailWindow` deltas and nonzero realized `window` fields are rejected.

## State declaration (`KVSTS1\0\0`)

The writer/ABI conformance object is:

```text
magic[8] || u16 version=1 || u16 zero || state_key ||
full_shape || segment_shape || u8 stride_count || strides ||
u64 logical_start || u64 logical_count || u64 absolute_position ||
u64 window || u32 atomic_group
```

It contains no caller-supplied schema root, chunk span, object ID, parent depth,
or path.

## Cut manifest (`KVMNF1\0\0`)

The canonical manifest is:

```text
magic[8]
u16 version=1
u16 zero
tenant_namespace Id32
key_epoch u64
SemanticModelId
InputCutId
family_body                    # embedded without KVFAM1 prefix/version
schema_body                    # embedded without KVRCS1 prefix/version
payload_state_count u32
payload_state[payload_state_count]
```

A payload state is `state_key || u32 chunk_count || chunk_ref[...]`. A chunk
reference is:

```text
chunk_content_id Id32
object_key Id32
stored_object_digest Id32
key_epoch u64
plaintext_bytes u32
object_bytes u32
```

Family, schema, and payload state counts and keys match exactly. Chunk counts
and plaintext sizes match schema spans exactly. Catalog epoch, prefix ancestors,
generation, location, supersession, tombstone state, continuation data, raw
tokens, and decorative roots are absent.
