# Identity and canonical scalar encoding

> **Pre-freeze draft.** No compatibility promise applies before Z1.

All integers are unsigned little-endian. `Id32` is exactly 32 bytes. A state key
is `u32 layer || u16 UTF-8-byte-length || name`; names are 1 through 255 bytes,
contain no NUL, and sort lexicographically by `(layer, raw UTF-8 bytes)`.

## Canonical identity objects

`SemanticModelId` is five consecutive `Id32` values: weights/config, adapters,
tokenizer/template, position semantics, and qualified math. Its public digest
is:

```text
SHA-256("kvpack/v1/semantic-model\0" || five fields)
```

`InputCutId` is `token_root Id32 || auxiliary_root Id32 || token_count u64`.
`token_count` is the only cut extent; zero is not a durable artifact.

The public family and realized-schema digests are SHA-256 over the respective
domain plus the complete standalone canonical object:

```text
SHA-256("kvpack/v1/representation-family\0" || KVFAM1 object)
SHA-256("kvpack/v1/realized-cut-schema\0" || KVRCS1 object)
```

The manifest ID is:

```text
SHA-256("kvpack/v1/manifest\0" || canonical KVMNF1 object)
```

## Auxiliary identities

Each auxiliary entry is `(type_id Id32, value_id Id32)`. For an ordered list,
form `u32 count`, then for each entry form
`u64 ordinal || u32 32 || type_id || u32 32 || value_id`. The root is:

```text
HMAC-SHA-256(prefix_key,
  "kvpack/v1/auxiliary-input-root\0" || tenant_namespace || framed_list)
```

Neither field can be zero. The list is ordered; duplicates are meaningful only
if the qualified semantic adapter says they are.

## Token prefix chain

First compute the family and semantic digests and the auxiliary root. Then:

```text
context = HMAC(prefix_key,
  "kvpack/v1/prefix-context\0" || tenant || semantic_digest ||
  family_digest || auxiliary_root)

parent_0 = HMAC(prefix_key, "kvpack/v1/prefix-root\0" || context)
```

Tokens are `u32` IDs and are divided into ordered blocks of at most 256. For
block `i`, starting at token `s`, encode:

```text
context || parent_i || u64(i) || u64(s) || u32(token_count) ||
u32(token_count * 4) || token_ids_as_little_endian_u32
```

The next node is HMAC under domain `kvpack/v1/prefix-node\0`. The final node and
exact token count form `InputCutId`; an empty sequence uses `parent_0` but is
recompute-only. Only nodes whose count is divisible by 256 are baseline reusable
checkpoints, although an exact final partial node may own a manifest.

## Chunk content identity

The stable tenant chunk-content key authenticates:

```text
"kvpack/v1/chunk-content\0" || tenant || family_digest ||
u32(state.layer) || u16(name_bytes) || state.name ||
u64(token_start) || u64(token_count) || u64(plaintext_offset) ||
u32(plaintext_bytes) || plaintext
```

`plaintext_offset` is the absolute logical byte offset from the beginning of
the complete state stream, including when the object first appears in a delta.
It therefore remains stable when a full manifest later reuses that object.

Codec and layout are already static family material. Key epoch, encryption,
nonce/salt, object key, and stored-object digest are deliberately not content
identity and are bound by the chunk object instead.

## Key separation

HKDF-SHA-256 with tenant salt derives stable namespace, prefix, and
chunk-content keys. A second HKDF salt `tenant || u64 key_epoch` derives
manifest-HMAC, manifest-AEAD, chunk-object, and chunk-AEAD keys. All labels use
the `kvpack/v1/stable/` or `kvpack/v1/epoch/` prefixes implemented by
`KeySchedule`; derived data-encryption keys are zeroized after use.
