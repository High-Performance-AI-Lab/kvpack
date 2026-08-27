# kvpack durable and live wire formats

**Status: DRAFT, pre-freeze.** Every byte layout, magic, and label in this
document remains mutable until the project freeze gate ("Z1"). Nothing here is
a compatibility promise. The normative sources remain
`spec/PACK_V1.md`, `spec/CHUNK_V1.md`, `spec/MANIFEST_V1.md`, and
`spec/IDENTITY_V1.md`; this document restates them for external evaluators and
cross-checks every offset against the reference Rust implementation
(`crates/kvpack-core`, `crates/kvpack`, and the `kvpack-handoff` crate that
implements the live wire). Where the prose specs and the code disagree, this
document follows the **code** and flags the drift explicitly (see section 10).

Audience: storage and systems engineers evaluating the format for integration
or audit. No site-specific configuration is described; every rule below is a
property of the bytes.

## Contents

1. Overview and object graph
2. Byte conventions
3. PACK_V1: the authenticated manifest container
4. CHUNK_V1: chunk objects and codec frames
5. MANIFEST_V1 and IDENTITY_V1: family, schema, manifest, identity chain
6. The live handoff wire (v1 and v2)
7. Publication and crash-safety model
8. Rejection order: the fail-closed matrix
9. Qualification status
10. Spec/code drift register

---

## 1. Overview and object graph

kvpack is an authenticated, crash-safe durable store for inference-engine
state — concretely, exact KV-cache prefixes keyed by a cryptographic identity
of model, input tokens, and cache geometry. It has two surfaces:

- **Durable objects.** Immutable files: one *pack file* carrying an
  authenticated, optionally encrypted *cut manifest*, plus any number of
  *chunk objects* carrying framed, optionally encrypted state bytes. A SQLite
  catalog (WAL, `synchronous=FULL`) is the sole authority on which objects are
  visible; it stores no identity material that the bytes do not already carry.
- **The live handoff wire.** A `KVHF`-framed stream of canonical-JSON
  manifests and raw KV planes used to move an exact prefix from a producer
  engine to a consumer, which then persists it through the same durable path.

The object graph at a glance:

```
+-------------------+-----------------------------------------------+
| pack file         |  4096-byte header (KVPKP1)                    |
|                   |  stored manifest (plaintext or AEAD + tag)    |
|                   |  4096-byte commit footer (KVCMT1, HMAC)       |
+-------------------+-----------------------------------------------+
                    | embeds (canonical bytes, KVMNF1)               
                    v                                                
+-------------------+-----------------------------------------------+
| cut manifest      |  tenant namespace, key epoch                  |
|                   |  SemanticModelId, InputCutId                  |
|                   |  representation family  (static inventory)    |
|                   |  realized cut schema    (shapes, spans)       |
|                   |  payload chunk references (ids, digests)      |
+-------------------+-----------------------------------------------+
                    | names chunks by content id + object key        
                    v                                                
+-------------------+-----------------------------------------------+
| chunk objects     |  4096-byte header (KVCHK1)                    |
|                   |  codec frame (KVRAW1 or KVRLE1) or AEAD+tag   |
|                   |  zero padding to a 4096-byte boundary         |
+-------------------+-----------------------------------------------+
```

A manifest is either a **full** base or a **delta** against a parent manifest.
Delta chains are bounded: one full base plus at most seven deltas.

```
+------------------------------------------------------------------+
| full (depth 0) <-- delta (depth 1) <-- ... <-- delta (depth 7)   |
|                                                                  |
| each delta links: parent manifest Id32 + parent InputCutId       |
| an attempted 8th append compacts to a new full manifest that     |
| re-references the same authenticated chunk objects unchanged     |
+------------------------------------------------------------------+
```

**Catalog vs object bytes.** Object files are content-addressed by digests
derived from their own bytes and are published no-replace. The catalog records
locations, refcounts, upload state, and prefix checkpoints — never trust. A
catalog row can point at bytes, but only the authenticated bytes prove what an
object is; recovery re-verifies objects against the catalog and quarantines
conflicts.

---

## 2. Byte conventions

These rules apply to every durable object (pack, chunk, and all canonical
identity objects). The live wire has its own conventions, stated in section 6.

- **Endianness.** All integers in durable objects are unsigned little-endian.
- **Widths.** Only `u8`, `u16`, `u32`, `u64` appear; there are no varints.
- **`Id32`.** A fixed 32-byte identifier (SHA-256 or HMAC-SHA-256 output).
- **Magics.** Eight bytes, ASCII with NUL padding, e.g. `KVPKP1\0\0`.
- **Canonical encoding.** Every count immediately precedes exactly that many
  entries. There are no optional sections, extension blocks, sentinel tails,
  or trailing bytes: a decoder must consume the final byte exactly, and every
  canonical object must **re-encode byte-identically** after decode or it is
  rejected.
- **Reserved bytes.** Every field marked "zero" must be zero and is checked.
  Nonzero reserved bytes are a hard error, not a hint.
- **Fail-closed versions and enums.** An unknown wire version, unknown enum
  value, unknown flag bit, or unknown magic is always a rejection. There is no
  "ignore and continue" path anywhere in the format.
- **Bounds.** Production caps enforced by the decoder (from
  `crates/kvpack-core/src/consts.rs`):

```
+--------------------------------------------+-----------------------+
| Bound                                      | Value                 |
+--------------------------------------------+-----------------------+
| alignment (all objects)                    | 4096 bytes            |
| chunk plaintext                            | 1 ..= 4 MiB           |
| codec frame header                         | 16 bytes              |
| max chunk object bytes (cap)               | 4,235,296 bytes       |
| max canonical manifest bytes               | 256 MiB               |
| max stats sidecar bytes (header tail)      | 3,858 bytes           |
| max states / atomic groups / chunks/state  | 65,536                |
| max dependencies per state                 | 64                    |
| state rank                                 | 1 ..= 8               |
| state name                                 | 1 ..= 255 UTF-8 bytes |
| delta chain depth                          | 0 ..= 7               |
| prefix chain block                         | 256 tokens            |
+--------------------------------------------+-----------------------+
```

The chunk-object cap is derived, not magic: `4096 (header) + 4 MiB (plaintext)
+ 32,784 (worst-case lossless overhead: 16-byte frame + one control byte per
128 bytes) + 16 (AEAD tag) + 4096 (one alignment quantum)`.

---

## 3. PACK_V1: the authenticated manifest container

A pack file is exactly:

```
4096-byte header  ||  stored manifest  ||  4096-byte commit footer
```

where the stored manifest is either the canonical plaintext manifest or, when
the AEAD flag is set, `ciphertext || 16-byte Poly1305 tag`. There is no record
stream, extension block, path, or catalog epoch anywhere in the file.

### 3.1 Header (4096 bytes)

```
+----------+------+----------------------------------------------------+
| Offset   | Size | Field / meaning                                    |
+----------+------+----------------------------------------------------+
| 0        | 8    | magic = "KVPKP1\0\0"                               |
| 8        | 2    | u16 wire version = 1                               |
| 10       | 2    | u16 header bytes = 4096                            |
| 12       | 4    | u32 alignment = 4096                               |
| 16       | 4    | u32 flags; bit 0 = manifest AEAD, others rejected  |
| 20       | 4    | u32 zero                                           |
| 24       | 8    | u64 stored manifest bytes (incl. AEAD tag)         |
| 32       | 8    | u64 plaintext canonical manifest bytes             |
| 40       | 8    | u64 manifest key epoch (nonzero)                   |
| 48       | 8    | u64 zero (catalog epoch is not durable identity)   |
| 56       | 32   | tenant namespace Id32                              |
| 88       | 32   | canonical manifest ID (plaintext identity)         |
| 120      | 16   | random AEAD salt; all zero when not encrypted      |
| 136      | 12   | random AEAD nonce; all zero when not encrypted     |
| 148      | 32   | SHA-256 header digest (zero while hashing)         |
| 180      | 3916 | zero                                               |
+----------+------+----------------------------------------------------+
```

The header digest is SHA-256 over the complete 4096-byte header with bytes
`148..180` treated as zero. It is written into the header before AEAD and HMAC,
so the digest itself is inside both authentications.

### 3.2 Body and AEAD

When the AEAD flag is set, the body is ChaCha20-Poly1305 over the canonical
manifest with the **exact final 4096-byte header as AAD**. The data key is:

```
HKDF-SHA-256( salt = header[120..136]  (random 16 bytes),                 
              ikm  = epoch manifest-encryption key,                       
              info = "kvpack/v1/manifest-aead\0" || manifest_id )  -> 32 B
```

with the random 12-byte nonce at `header[136..148]`. Without AEAD, salt and
nonce must be zero (checked) and the body is the plaintext manifest. Stored
bytes therefore equal `plaintext_bytes + 16` with AEAD, `plaintext_bytes`
without. Derived data keys are zeroized after use.

### 3.3 Commit footer (4096 bytes)

```
+----------+------+----------------------------------------------------+
| Offset   | Size | Field / meaning                                    |
+----------+------+----------------------------------------------------+
| 0        | 8    | magic = "KVCMT1\0\0"                               |
| 8        | 2    | u16 wire version = 1                               |
| 10       | 2    | u16 footer bytes = 4096                            |
| 12       | 4    | u32 zero                                           |
| 16       | 8    | u64 exact stored manifest bytes                    |
| 24       | 8    | u64 exact manifest key epoch                       |
| 32       | 8    | u64 zero                                           |
| 40       | 8    | u64 exact complete file bytes                      |
| 48       | 32   | exact manifest ID                                  |
| 80       | 32   | HMAC-SHA-256 (see 3.4)                             |
| 112      | 3984 | zero                                               |
+----------+------+----------------------------------------------------+
```

The footer is the commit record: it restates the stored length, key epoch,
manifest ID, and complete file length, and authenticates all of it.

### 3.4 HMAC coverage

The footer HMAC-SHA-256 is computed under the epoch manifest-auth key over the
following bytes, in exactly this order:

```
+------------------------------------------------------------------+
| "kvpack/v1/manifest-auth\0"            (24-byte domain)          |
+------------------------------------------------------------------+
| complete 4096-byte header (digest field already filled)          |
+------------------------------------------------------------------+
| stored manifest body (plaintext, or ciphertext || tag)           |
+------------------------------------------------------------------+
| u64 stored manifest bytes                                        |
+------------------------------------------------------------------+
| u64 manifest key epoch                                           |
+------------------------------------------------------------------+
| u64 complete file bytes                                          |
+------------------------------------------------------------------+
                                  |                                 
                                  v                                 
                     footer[80..112]  (32 bytes)                    
```

Everything an attacker could flip — header fields, lengths, epoch, file size,
ciphertext — is inside the HMAC. The plaintext manifest ID and plaintext length
are additionally rechecked after decryption/canonicalization, and the decoded
manifest's tenant and key epoch must equal the header's.

### 3.5 Decode order (fail-closed)

`decode_authenticated_pack` (`crates/kvpack-core/src/pack.rs`) rejects in this
order:

1. Minimum size (`>= 8192`); header magic/version/sizes; unknown flags;
   all reserved bytes zero; salt/nonce zero when not encrypted.
2. Manifest length bounds (`stored <= 256 MiB + 16`, `plaintext <= 256 MiB`).
3. Footer magic/version/reserved; exact footer↔header linkage (stored length,
   epoch, file length, manifest ID).
4. Header digest; then footer HMAC; then AEAD (if flagged).
5. Plaintext length and manifest ID equality.
6. Full canonical `KVMNF1` decode **and byte-identical re-encode**.
7. Header/manifest tenant and epoch equality; semantic/graph validation
   (`validate_manifest`); parent-chain and chunk validation are later phases.

Every development-era magic (`IOKVPK1`, `IOKVENC`, zlib/Q8 drafts, record
magics) is simply an unknown magic and fails at step 1/3.

---

## 4. CHUNK_V1: chunk objects and codec frames

A chunk object is:

```
4096-byte header  ||  framed payload (or ciphertext || tag)  ||  zero padding
```

padded with zero bytes to a 4096-byte boundary. Plaintext is 1 through 4 MiB
and always contains an integral number of logical state tokens. The
authenticated manifest supplies the expected family, state key, token/byte
span, content ID, object key, stored-object digest, and key epoch; the chunk
decoder checks all of them.

### 4.1 Header (4096 bytes)

```
+----------+------+----------------------------------------------------+
| Offset   | Size | Field / meaning                                    |
+----------+------+----------------------------------------------------+
| 0        | 8    | magic = "KVCHK1\0\0"                               |
| 8        | 2    | u16 wire version = 1                               |
| 10       | 2    | u16 header bytes = 4096                            |
| 12       | 4    | u32 alignment = 4096                               |
| 16       | 4    | u32 flags; bit 0 = AEAD, no other bit accepted     |
| 20       | 2    | u16 codec: raw = 1, lossless = 2                   |
| 22       | 2    | u16 codec version = exactly 1                      |
| 24       | 4    | u32 decoded plaintext bytes                        |
| 28       | 4    | u32 complete codec-frame bytes (before AEAD tag)   |
| 32       | 4    | u32 stored payload bytes (incl. tag if encrypted)  |
| 36       | 4    | u32 complete aligned object bytes                  |
| 40       | 8    | u64 chunk-object key epoch (nonzero)               |
| 48       | 32   | tenant namespace Id32                              |
| 80       | 32   | static representation-family digest                |
| 112      | 32   | keyed plaintext chunk-content ID                   |
| 144      | 32   | epoch-specific object key (see 5.7)                |
| 176      | 16   | random salt; zero without AEAD                     |
| 192      | 12   | random nonce; zero without AEAD                    |
| 204      | 32   | SHA-256 header digest (zero while hashing)         |
| 236      | 3860 | stats sidecar tail: u16 sidecar length, then the   |
|          |      | canonical sidecar bytes, then zero padding to      |
|          |      | 4096; all zero when no sidecar is attached (4.3)   |
+----------+------+----------------------------------------------------+
```

The stored-object digest (SHA-256 over the complete object: header, stored
payload, tag, and padding) is supplied by the authenticated manifest and is
checked after all framing/reserved checks and before any semantic or plaintext
use. The header digest is SHA-256 over the header with bytes `204..236` zeroed
— so the sidecar tail is inside the header digest, the stored-object digest,
and (when encrypted) the AEAD AAD.

**Sidecar presence.** The tail at offset 236
(`CHUNK_HEADER_SIDECAR_OFFSET` in `consts.rs`) carries an optional statistics
sidecar: a nonzero u16 length prefix means a canonical sidecar follows (4.3);
zero means the whole 3860-byte tail must be zero. Presence deliberately
consumes **no flag bit** — `KNOWN_FLAGS` stays `1` (bit 0 = AEAD) — so a
pre-sidecar reader validates the tail as reserved-zero and rejects a
sidecar-carrying object as non-canonical reserved bytes, and a sidecar-absent
object is byte-identical to the pre-sidecar format (same object key, same
digests).

### 4.2 Codec frames

Every chunk is independently framed, including raw. Both frame headers are
16 bytes:

```
+----------+------+----------------------------------------------------+
| Offset   | Size | Field / meaning                                    |
+----------+------+----------------------------------------------------+
| 0        | 8    | frame magic: "KVRAW1\0\0" or "KVRLE1\0\0"          |
| 8        | 2    | u16 frame version = 1                              |
| 10       | 2    | u16 zero                                           |
| 12       | 4    | u32 decoded plaintext bytes                        |
+----------+------+----------------------------------------------------+
```

**Raw (codec 1).** The frame header followed by exactly `decoded_bytes` of
payload. Nothing else.

**Lossless (codec 2).** The frame header followed by canonical PackBits-style
packets. Each packet begins with a control byte; `length = (control & 0x7f)+1`.

```
+------------------------------------------------------------------+
| control bit 7 = 0 (literal): exactly `length` payload bytes      |
| control bit 7 = 1 (repeat):  one payload byte, repeated `length` |
+------------------------------------------------------------------+
```

The packetization is canonical, not merely valid: the encoder emits a repeat
packet exactly for runs of at least three equal bytes (maximum 128 per packet)
and joins all other bytes into the longest literal packet (maximum 128) that
ends before such a run. A decoder must produce exactly the declared size and
then **byte-identically re-encode the frame**, rejecting alternate
packetizations of the same plaintext. Worst-case lossless expansion is bounded
by the 16-byte frame header plus one control byte per 128 plaintext bytes
(`16 + ceil(n/128)`).

**Bound checks (decode order).** Object length aligned to 4096 and within cap;
magic/version/sizes; unknown flags; codec enum; codec version; `plaintext` in
`1..=4 MiB`; `encoded` in `16..=4 MiB + 32,784`; declared object bytes equal
actual file length; `payload == encoded + (AEAD ? 16 : 0)`; padding zero;
reserved header bytes zero; salt/nonce zero when not encrypted. Only then come
the stored-object digest, header digest, family codec match, tenant/family/
epoch binding, object-key recomputation, AEAD, canonical codec decode and
re-encode, and finally the plaintext content-ID check.

**AEAD.** ChaCha20-Poly1305 with the exact final 4096-byte header as AAD. Data
key:

```
HKDF-SHA-256( salt = header[176..192]  (random 16 bytes),                  
              ikm  = epoch chunk-encryption key,                           
              info = "kvpack/v1/chunk-aead\0" || content_id || object_key )
```

**Closed codec set.** Raw (1) and lossless (2) are the only production codec
values. Q8 has no production frame; Q4/Q2, zlib draft frames, and every other
codec number or frame magic are unknown values and fail closed at the enum or
frame-magic check.

### 4.3 Statistics sidecar (`KVSSC1\0\0`)

The optional sidecar (`crates/kvpack-core/src/stats.rs`) is a canonical,
authenticated record of per-cut attention statistics derived from a fp16
K-state plane: the per-channel K min/max a fidelity-rung re-encode would need
as quantization scales, the per-token key L2 norms, and the bounded top-m
attention-sink scores. It is derived from a `tokens × channels` token-major
binary16 plane and fails closed on any non-finite element. Bounds:
channels `1..=512`, tokens `1..=768`, sinks `1..=8` (`MAX_SIDECAR_CHANNELS`,
`MAX_SIDECAR_TOKENS`, `MAX_SINK_SCORES`), and the canonical encoding must fit
the header tail (`MAX_STATS_SIDECAR_BYTES = 4096 − 236 − 2 = 3,858`).

Canonical layout (all integers little-endian):

```
+----------+------+----------------------------------------------------+
| Offset   | Size | Field / meaning                                    |
+----------+------+----------------------------------------------------+
| 0        | 8    | magic = "KVSSC1\0\0"                               |
| 8        | 2    | u16 version = 1                                    |
| 10       | 2    | u16 zero                                           |
| 12       | 4    | u32 channel count (1..=512), then per channel:     |
| 16       | 4*ch | u16 min bits || u16 max bits (binary16), in        |
|          |      | channel order; finite and min <= max               |
| ...      | 4    | u32 token count (1..=768), then per token:         |
| ...      | 2*tk | u16 L2-norm bits (binary16), in token order;       |
|          |      | finite and non-negative                            |
| ...      | 2    | u16 sink count (1..=8), then per sink:             |
| ...      | 8*sk | u32 token index || u16 score bits || u16 zero      |
+----------+------+----------------------------------------------------+
```

The sink list must be exactly the top-m of the carried norms — descending
score, ties broken by ascending token index — so a forged or reordered
ranking is malformed, not merely stale. Decode validates every bound and
ordering rule and requires a byte-identical re-encode, like every other
canonical object.

**Identity binding.** `stats_digest = SHA-256(canonical sidecar bytes)` is
mixed into the object key exactly like the other authenticated header fields:
when a sidecar is present, the object-key HMAC input (5.7) gains
`"kvpack/v1/chunk-object-stats\0" || stats_digest`; when absent it gains
nothing, so pre-sidecar object keys are unchanged. The sidecar bytes
themselves sit inside the header digest, the stored-object digest, and the
AEAD AAD, so they are integrity-protected by the same authentications as the
rest of the header.

---

## 5. MANIFEST_V1 and IDENTITY_V1

All canonical identity objects share the envelope `magic[8] || u16 version=1
|| u16 zero || body` and the re-encode-exactly rule. Counts precede their
entries; there are no optional sections.

### 5.1 Representation family (`KVFAM1\0\0`)

The family is the static compatibility contract for one engine cache ABI: what
states exist, their dtype/codec/layout, and the token axis. It deliberately
contains no token extents, shapes, chunk spans, or identities of a specific
cut.

Family body:

```
+----------+------+----------------------------------------------------+
| Field    | Size | Meaning                                            |
+----------+------+----------------------------------------------------+
| engine_cache_abi | 32 | Id32, nonzero                                |
| representation_mode | 2 | u16: native = 1, portable = 2              |
| zero     | 2    | u16 zero                                           |
| page_size_tokens | 4 | u32, nonzero                                  |
| topology | 32   | Id32, nonzero                                      |
| shard_map | 32  | Id32, nonzero                                      |
| state_count | 4 | u32, 1..=65,536, then that many states             |
+----------+------+----------------------------------------------------+
```

Each static state:

```
+----------+------+----------------------------------------------------+
| Field    | Size | Meaning                                            |
+----------+------+----------------------------------------------------+
| state_key | 6+n | u32 layer || u16 name bytes || UTF-8 name (1..255, |
|          |      | no NUL; sorted by (layer, raw name bytes))         |
| cache_kind | 2  | u16: ordinary KV = 1 (only value)                  |
| dtype    | 2    | u16 DType (table below)                            |
| codec    | 2    | u16: raw = 1, lossless = 2                         |
| codec_version | 2 | u16, exactly 1                                   |
| layout   | 2    | u16: contiguous = 1, strided = 2                   |
| token_axis_rule | 2 | u16: direct = 1, gather = 2, TailWindow = 3    |
| token_axis | 1  | u8 index of the token dimension                    |
| zero     | 1    | u8 zero                                            |
| elements_per_token | 8 | u64, checked product of fixed dims          |
| rank     | 1    | u8, 1..=8                                          |
| dimension | 8*rank | u64 each; 0 = token, fixed dims nonzero         |
| dependency_count | 2 | u16, <= 64, then that many state_keys         |
+----------+------+----------------------------------------------------+
```

Exactly one dimension is zero (the token extent) and its index equals
`token_axis`. States and dependency lists are in unique canonical order and
the dependency graph is closed and acyclic. A contiguous state cannot declare
`gather`. Portable mode requires contiguous layout and a `direct` or
`TailWindow` token-axis rule (see drift note D2: the prose spec still says
"contiguous/direct" and omits `TailWindow = 3`; the code defines and admits
it).

`TailWindow` states model sliding-window attention layers: a full snapshot
carries only the trailing in-window tokens of the prefix. Both realized shapes
use the compact stored tail extent, `logical_start` retains its absolute token
position, and chunk plaintext offsets remain absolute. TailWindow deltas are
rejected. The realized-state `window` field itself remains zero (drift note
D3); windowing is expressed through the family rule and authenticated range.

DType wire values (fixed-width; width in bytes):

```
+-------+-----------+-------+-----------+-------+-----------+
| value | type      | bytes | value     | bytes |           |
+-------+-----------+-------+-----------+-------+-----------+
| 1     | u8        | 1     | 6  f16    | 2     |           |
| 2     | u32       | 4     | 7  bf16   | 2     |           |
| 3     | i8        | 1     | 8  f32    | 4     |           |
| 4     | i16       | 2     | 9  f64    | 8     |           |
| 5     | i32       | 4     |           |       |           |
+-------+-----------+-------+-----------+-------+-----------+
```

### 5.2 Realized cut schema (`KVRCS1\0\0`)

The schema pins one exact cut to concrete shapes, ranges, strides, and chunk
spans. The body begins with the manifest-kind union:

```
+------------------------------------------------------------------+
| kind u8 || depth u8 || zero u16                                  |
|   kind = 0 (full):  depth = 0, no union payload                  |
|   kind = 1 (delta): depth = 1..=7, then                          |
|                     parent_manifest Id32 || parent InputCutId    |
|   all other tags: unknown, fail closed                           |
+------------------------------------------------------------------+
```

Then `u32 state_count` realized states, `u32 atomic_group_count` groups, and
two `u64` totals (`segment_restored_bytes`, `complete_restored_bytes`). Each
realized state:

```
+----------+------+----------------------------------------------------+
| Field    | Size | Meaning                                            |
+----------+------+----------------------------------------------------+
| state_key | 6+n | as in the family                                   |
| full_shape | 1+8r | direct: child cut; TailWindow: stored tail count |
| segment_shape | 1+8r | same, with covered range count at axis        |
| stride_count | 1 | u8, then that many u64 element strides            |
| logical_start | 8 | direct cut start or TailWindow absolute start   |
| logical_count | 8 | u64; covered token count                         |
| physical_offset_bytes | 8 | u64, checked footprint                   |
| physical_span_bytes | 8 | u64, checked footprint                     |
| complete_physical_bytes | 8 | u64, checked footprint                 |
| absolute_position | 8 | u64, equals the child cut token count        |
| window   | 8    | u64; must be zero in the current code (see D3)     |
| chunk_span_count | 4 | u32, 1..=65,536, then that many spans         |
+----------+------+----------------------------------------------------+
```

A chunk span is `u64 token_start || u64 token_count || u64 plaintext_offset
|| u32 plaintext_bytes`. Spans partition the covered token and byte ranges
with no gaps, overlaps, empty entries, or token splits; `plaintext_bytes`
equals `token_count` times the family's exact bytes per token and is at most
4 MiB. `plaintext_offset` is **absolute from the start of the complete logical
state stream**, not relative to a delta segment — this is what lets the same
authenticated chunk object appear unchanged in a delta and in a later
reference-only full compaction.

A full state covers `[0, child_cut)`; a delta state covers exactly
`[parent_cut, child_cut)`. An atomic group is `u32 nonzero_id || u32
state_count || state_key[...]`; groups sort by ID, members sort canonically,
and together they partition the complete family inventory exactly once.

### 5.3 State declaration (`KVSTS1\0\0`)

The writer/ABI conformance object: the schema header, then `state_key`,
`full_shape`, `segment_shape`, `u8 stride_count`, strides, `u64
logical_start`, `u64 logical_count`, `u64 absolute_position`, `u64 window`,
`u32 atomic_group`. It contains no chunk spans, object IDs, parent links, or
paths.

### 5.4 Cut manifest (`KVMNF1\0\0`)

```
+----------+------+----------------------------------------------------+
| Field    | Size | Meaning                                            |
+----------+------+----------------------------------------------------+
| magic/version/zero | 12 | "KVMNF1\0\0" || u16 1 || u16 0             |
| tenant_namespace | 32 | Id32                                         |
| key_epoch | 8  | u64, nonzero                                        |
| SemanticModelId | 160 | five Id32 (see 5.5)                          |
| InputCutId | 72 | token_root || auxiliary_root || u64 token_count    |
| family_body | var | family body, embedded WITHOUT the KVFAM1 prefix  |
| schema_body | var | schema body, embedded WITHOUT the KVRCS1 prefix  |
| payload_state_count | 4 | u32, then that many payload states         |
+----------+------+----------------------------------------------------+
```

A payload state is `state_key || u32 chunk_count || chunk_ref[...]`, where a
chunk reference is:

```
+----------+------+----------------------------------------------------+
| Field    | Size | Meaning                                            |
+----------+------+----------------------------------------------------+
| chunk_content_id | 32 | keyed plaintext content ID                   |
| object_key | 32 | epoch-specific object key                          |
| stored_object_digest | 32 | SHA-256 of the complete chunk object     |
| key_epoch | 8  | u64, nonzero                                        |
| plaintext_bytes | 4 | u32, must equal the schema span                |
| object_bytes | 4 | u32, 4096-aligned, header < size <= cap           |
+----------+------+----------------------------------------------------+
```

Family, schema, and payload state counts and keys match exactly, in canonical
order; chunk counts and plaintext sizes match the schema spans exactly.
Catalog epochs, prefix ancestors, generations, locations, supersession,
tombstones, and raw tokens are deliberately absent from the durable identity.

### 5.5 Semantic model identity

`SemanticModelId` is five consecutive `Id32` values: weights/config, adapters,
tokenizer/template, position semantics, and qualified math. Its public digest:

```
SHA-256( "kvpack/v1/semantic-model\0" || five Id32 fields )
```

The tenant namespace is derived, not stored as raw configuration:

```
HMAC-SHA-256( namespace_key,                                            
              "kvpack/v1/namespace\0" || u32 len || operator tenant id )
```

### 5.6 The keyed identity chain

Public digests of the standalone objects:

```
SHA-256( "kvpack/v1/representation-family\0" || KVFAM1 object )          
SHA-256( "kvpack/v1/realized-cut-schema\0"   || KVRCS1 object )          
SHA-256( "kvpack/v1/manifest\0"              || canonical KVMNF1 object )
```

The input-cut identity is a keyed HMAC chain over the exact token prefix, in
blocks of at most 256 tokens (`u32` token IDs, little-endian):

```
prefix_key (HKDF stable/prefix, per tenant)                           
   |                                                                  
   |  aux_root = HMAC(key, "kvpack/v1/auxiliary-input-root\0" ||      
   |                    tenant || u32 count ||                        
   |                    per entry: u64 ordinal || u32 32 || type_id ||
   |                               u32 32 || value_id)                
   v                                                                  
 context = HMAC(key, "kvpack/v1/prefix-context\0" || tenant ||        
                semantic_digest || family_digest || aux_root)         
   |                                                                  
 parent0 = HMAC(key, "kvpack/v1/prefix-root\0" || context)            
   |                                                                  
   |  block i starting at token s:                                    
   |  node(i+1) = HMAC(key, "kvpack/v1/prefix-node\0" || context ||   
   |                    node(i) || u64(i) || u64(s) ||                
   |                    u32(count) || u32(count*4) || tokens LE u32)  
   v                                                                  
 final node = token_root                                              
   |                                                                  
 InputCutId = token_root || aux_root || u64 token_count               
```

Only nodes whose cumulative count is divisible by 256 are baseline-reusable
checkpoints; an exact final partial node may own a manifest but is
recompute-only. An empty token sequence reduces to `parent0` and is not a
durable artifact (`token_count = 0` is invalid). No token witness is ever
retained in a manifest or catalog row.

The chunk content ID authenticates plaintext under the stable chunk-content
key:

```
HMAC-SHA-256( chunk_identity_key,                                 
  "kvpack/v1/chunk-content\0" || tenant || family_digest ||       
  u32(state.layer) || u16(name bytes) || state.name ||            
  u64(token_start) || u64(token_count) || u64(plaintext_offset) ||
  u32(plaintext_bytes) || plaintext )                             
```

Codec and layout are static family material; key epoch, encryption, salt,
nonce, object key, and stored-object digest are deliberately **not** content
identity — they are bound by the chunk object instead.

### 5.7 The object key

The epoch-specific object key distinguishes one stored representation of a
chunk from any other (e.g. re-encryption under a new epoch):

```
HMAC-SHA-256( object_identity_key,                                      
  "kvpack/v1/chunk-object\0" || tenant || family_digest || content_id ||
  u64 key_epoch || u16 codec || u16 codec_version || u8 encrypted ||    
  salt[16] || nonce[12] ||                                              
  [ if a stats sidecar is present:                                      
    "kvpack/v1/chunk-object-stats\0" || stats_digest ] )                
```

`stats_digest` is SHA-256 over the canonical sidecar bytes (4.3). An absent
sidecar contributes nothing, so the HMAC input — and therefore the object key
— is byte-identical to the pre-sidecar derivation for sidecar-absent objects.

### 5.8 Key separation

One 32-byte stable root and per-epoch 32-byte roots derive all working keys
via HKDF-SHA-256; epoch 0 is invalid; the readable window is bounded (active
minus minimum readable < 64):

```
+------------------------------------------------------------------+
| stable HKDF (salt = tenant Id32, ikm = stable root)              |
|   "kvpack/v1/stable/namespace"       -> namespace key            |
|   "kvpack/v1/stable/prefix"          -> prefix key               |
|   "kvpack/v1/stable/chunk-identity"  -> chunk content key        |
+------------------------------------------------------------------+
| epoch HKDF (salt = tenant Id32 || u64 LE epoch, ikm = epoch root)|
|   "kvpack/v1/epoch/manifest-auth"        -> pack HMAC key        |
|   "kvpack/v1/epoch/manifest-encryption"  -> manifest AEAD root   |
|   "kvpack/v1/epoch/object-identity"      -> object key root      |
|   "kvpack/v1/epoch/chunk-encryption"     -> chunk AEAD root      |
+------------------------------------------------------------------+
```

Stable identities (namespace, prefix chain, chunk content) survive key
rotation; everything epoch-scoped can be retired by tombstoning objects below
a minimum readable epoch and garbage-collecting the bytes. All key material
is zeroized on drop.

### 5.9 Transform descriptors (`KVXFM1\0\0`)

A `TransformDescriptor` (`crates/kvpack-core/src/transform.rs`, M4) is an
authenticated engine-ABI translation: an ordered list of repack ops mapping a
named source layout class onto the canonical layout. The canonical binary
encoding is the only authenticated representation — `transform_id` is
SHA-256 over the canonical bytes — and the JSON form (`deny_unknown_fields`,
fail-closed) is a control-plane representation, never an identity.

Canonical layout (all integers little-endian; unlike the other canonical
objects there is **no** reserved u16 after the version):

```
+----------+------+----------------------------------------------------+
| Field    | Size | Meaning                                            |
+----------+------+----------------------------------------------------+
| magic    | 8    | "KVXFM1\0\0"                                       |
| version  | 2    | u16, exactly 1                                     |
| name     | 2+n  | u16 byte count || ASCII label (1..=128 bytes)      |
| source_layout | 2+n | u16 byte count || ASCII label (1..=128)        |
| op_count | 2    | u16, 0..=64, then that many ops                    |
+----------+------+----------------------------------------------------+
```

Op encoding: a u16 tag followed by tag-specific parameters.

```
+-----+-------------------+--------------------------------------------+
| tag | op                | parameters                                 |
+-----+-------------------+--------------------------------------------+
| 1   | permute-head-pairs| u16 width || width x u32 order (perm of    |
| 2   | reorder-planes    | 0..width, 1..=16,384 entries); same wire   |
| 3   | regroup-layers    | form for all three permutation ops         |
| 4   | pad-or-trim       | u64 target_bytes (1..=2^30)                |
| 5   | dtype-cast        | u16 from || u16 to (1 fp16, 2 bf16, 3      |
|     |                   | fp8e4m3) || u8 scale-id present || Id32    |
|     |                   | (zero when absent)                         |
| 6   | rope-permute      | u16 direction (1 neox->interleaved, 2      |
|     |                   | interleaved->neox) || u32 head_dim (even,  |
|     |                   | 2..=4,096)                                 |
+-----+-------------------+--------------------------------------------+
```

`order[i]` is the source index placed at output slot `i`; a permutation must
be non-empty, in-range, and repeat-free. A `dtype-cast` must name distinct
endpoints, and a scale-id is present exactly when an endpoint is `fp8e4m3`.
Five of the six ops are exact index permutations or byte padding and are
executable (including inverse); **`dtype-cast` is named-only** — it is not
bit-exact and the executor rejects it, so it may document a lane but never
runs. Unknown tags, dtypes, directions, or out-of-bounds parameters fail
closed at decode/validate.

**Identity binding.** The v2 prefill descriptor input carries
`transform: Option<Id32>` — the canonical descriptor's `transform_id`. It
binds into the derived representation family through the
engine-cache-abi/v3 hash (5.10): present contributes the 32-byte id, absent
contributes zero bytes, so transformed and untransformed artifacts can never
collide.

### 5.9a Rotation-family descriptors (`KVRTA1\0\0`)

A `RotationFamilyDescriptorV1` (`crates/kvpack-core/src/rotation.rs`) reserves
the authenticated identity boundary for full-precision positional rotation.
It is identity material only: the V1 reservation does not execute rotation or
enable a runtime route. The complete contract, including fixed operation and
cast order, is in `docs/design/ROTATION-ABI-V1.md`.

Canonical layout (all integers little-endian):

```
+--------------------------+------+-------------------------------------+
| Field                    | Size | Meaning                             |
+--------------------------+------+-------------------------------------+
| magic                    | 8    | "KVRTA1\0\0"                        |
| version                  | 2    | u16, exactly 1                      |
| reserved                 | 2    | u16, exactly 0                      |
| coefficient_set          | 2    | u16 enum                            |
| sincos_order             | 2    | u16 enum                            |
| phase_origin             | 2    | u16 enum                            |
| position_convention      | 2    | u16 enum                            |
| pairing                  | 2    | u16 enum                            |
| denormal_policy          | 2    | u16 enum                            |
| f32_rounding             | 2    | u16 enum                            |
| f16_cache_cast           | 2    | u16 enum                            |
| rotation_order           | 2    | u16 enum                            |
| reserved                 | 2    | u16, exactly 0                      |
| rotary_dimension         | 4    | even u32, 2..=4,096                 |
| coefficient_set_sha256   | 32   | authenticated coefficient asset    |
| model_representation_id  | 32   | nonzero model/representation Id32  |
| frac64_len               | 4    | u32; exactly rotary_dimension/2*8  |
| frac64                   | n    | complete u64 LE increments         |
| frac64_sha256            | 32   | SHA-256 over exact frac64 bytes    |
+--------------------------+------+-------------------------------------+
```

All currently defined enums use wire value 1; unknown values, nonzero
reserved fields, zero identities/increments, table-size/hash disagreement,
and a table whose every low 32-bit half is zero fail closed. Descriptor
identity is `SHA-256("kvpack/v1/rotation-family\0" || canonical_bytes)`.
Binding uses the `kvpack/v1/rotation-bound-engine-cache-abi\0` domain and
changes only `RepresentationFamilyId.engine_cache_abi`. An absent hook returns
the original family byte-for-byte.

### 5.10 Prefill v2 identity derivation (engine-cache-abi/v3)

The v2 prefill lane (`crates/kvpack/src/prefill/v2.rs`) derives the
representation family from the validated BEGIN fields. Every derived Id32 is

```
domain_id(domain, parts) =                                               
  SHA-256( domain || per part: u64 LE part length || part )              
```

The semantic model fields use `kvpack/spark-prefill/{weights-config,
adapters, tokenizer-template, position-semantics, qualified-math}/v1\0`
domains; the engine cache ABI uses:

```
engine_cache_abi =                                                       
  domain_id( "kvpack/spark-prefill/engine-cache-abi/v3\0",              
             portable_abi || consumer_engine_abi ||                      
             labeled_geometry || transform_binding )                     
```

where `transform_binding` is the 32-byte `transform_id` or empty (5.9), and
`labeled_geometry` serializes each class **in declared table order** as:

```
+------------------------------------------------------------------+
| per class:                                                       |
|   u32 from || u32 until || u32 step || u32 kv_heads ||           |
|   u32 head_dim || u32 window_tokens || u32 except[...]           |
|   u32 class-label bytes || class label (ASCII)                   |
|   u32 state-name count || per name: u32 bytes || name            |
| then: u32 max_context_tokens                                     |
+------------------------------------------------------------------+
```

The state-name list comes from `class_state_names`: `["attn.kv_latent"]`
for an `mla-latent` class, `["attn.k", "attn.v"]` otherwise — the same
derivation that emits the descriptor states, so identity and state emission
cannot drift apart. v3 supersedes v2 (which hashed the bare numeric geometry
only, so same-geometry layouts of different classes collided): the class
label and state names bind the layout *semantics* into the identity. This is
a deliberate identity break — all v2-derived `engine_cache_abi` values
change — recorded as drift D5; v2 is retained for the existing qualified
lanes as a test fixture (`legacy_v2_engine_cache_abi`).

---

## 6. The live handoff wire (v1 and v2)

The live wire moves one exact KV prefix from a producer engine to a consumer
before persistence. It is specified by `spec/KVPACK_LIVE_V2.md` (pinned wire
contract, additive over v1) and implemented by the `kvpack-handoff` crate.
Protocol label `kvpack-live-f16-le-v1`, schema version 1, for both v1 and v2;
v2 is signaled inside the BEGIN manifest, not by a new frame version.

**Note on the envelope:** the frame magic is `KVHF` (4 bytes). The prose spec
describes the envelope only as "unchanged from v1"; the field sizes and the
big-endian integer encoding below come from the implementation
(`crates/kvpack-handoff/src/frame.rs`) and are not spelled out in the spec
text. The durable objects are little-endian; this envelope is not.

### 6.1 Frame envelope (20-byte header, big-endian)

```
+----------+------+----------------------------------------------------+
| Offset   | Size | Field / meaning                                    |
+----------+------+----------------------------------------------------+
| 0        | 4    | magic = "KVHF"                                     |
| 4        | 1    | u8 schema version = 1                              |
| 5        | 1    | u8 kind: 1 BEGIN, 2 LAYER, 3 SEAL, 4 ABORT, 5 ACK  |
| 6        | 2    | u16 reserved = 0 (big-endian)                      |
| 8        | 4    | u32 JSON manifest bytes (big-endian), 1..=1 MiB    |
| 12       | 8    | u64 payload bytes (big-endian), 0..=64 MiB         |
+----------+------+----------------------------------------------------+
```

The header is followed by the JSON manifest and then the payload. Only LAYER
frames may carry a payload; for every other kind `payload_len` must be 0, and
for LAYER frames `payload_len` must equal the header's declared `byte_length`.
Unknown kind, version, magic, or nonzero reserved bytes fail before any
allocation. The receiver decodes the envelope and JSON first and only then
acquires a bounded memory permit for the payload; the reader refuses a second
envelope while a payload is pending.

### 6.2 Canonical JSON

Every manifest is canonical JSON:

- keys sorted lexicographically (Rust serializes through `serde_json::Value`,
  Python through `sort_keys=True` — byte-identical),
- compact separators (no insignificant whitespace),
- no unknown fields (`deny_unknown_fields` on every struct),
- integers only in these schemas (no floats),
- a decoder must re-encode the parsed value byte-identically or reject,
- length 1..=1 MiB per frame manifest.

All SHA-256 values are 64 lowercase hex characters; all label strings are
printable ASCII (no `\`, `/`, `"`), bounded length (64 for class labels, 256
for endpoint/revision labels).

### 6.3 Stream shape

```
+------------------------------------------------------------------+
| producer                                   consumer              |
|    |                                            |                |
|    |-- BEGIN (identity, geometry, layout) ----->|  validate      |
|    |-- LAYER seq 0 (header JSON || plane) ----->|  per-plane chk |
|    |-- LAYER seq 1 ---------------------------->|     ...        |
|    |        ... (expected_layer_frames total)   |                |
|    |-- SEAL (chain hash, artifact seal) ------->|  verify, commit|
|    |<------------------------------ ACK ------ |  "committed"    |
|    |-- ABORT (code) --------------------------->|  either side   |
+------------------------------------------------------------------+
```

### 6.4 BEGIN (v1 fields, all required in v1 and v2)

```
+----------------------------+-----------------------------------------+
| Field                      | Meaning                                 |
+----------------------------+-----------------------------------------+
| cached_token_count         | tokens cached, 1..=32,767               |
| created_unix_ms            | creation time (skew-bounded)            |
| deadline_unix_ms           | session deadline (<= 15 min)            |
| endpoints                  | 5 labels: producer/consumer engine ABI, |
|                            | producer/consumer node, trust domain    |
| expected_layer_frames      | must equal the walk length              |
| expected_payload_bytes     | must equal the walk's byte count        |
| geometry                   | num_layers, num_kv_heads, head_dim,     |
|                            | max_context_tokens                      |
| identity                   | model/tokenizer/adapter/template/policy |
|                            | sha256 + revisions (exact match)        |
| portable_abi               | "canonical-kv-f16-le-v1" (v1) or        |
|                            | "canonical-kv-v2" (non-empty table)     |
| precision                  | compute = float16, kv = float16,        |
|                            | weights: closed set (see D1)            |
| protocol                   | "kvpack-live-f16-le-v1"                 |
| schema_version             | 1                                       |
| strategy                   | "consumer_last_prompt_token"            |
| token_ids_sha256           | over "kvpack-live-token-ids-v1\0" ||    |
|                            | LE u32 token ids                        |
| transfer_id                | 64-char lowercase hex                   |
| layout_table               | v2 only; absent/empty = v1 semantics    |
| schedule                   | v2 only; absent = "layer-order" (6.7)   |
+----------------------------+-----------------------------------------+
```

`layout_table` is omitted from the JSON when empty, so a v1 BEGIN is byte- and
hash-identical under v2 rules. When it is non-empty, `portable_abi` must be
`canonical-kv-v2`. `schedule` is likewise omitted when absent, so a
pre-schedule v2 BEGIN stays byte- and hash-identical; a v1 BEGIN (empty
`layout_table`) may not carry it at all.

### 6.5 v2 layout classes

Each class names a set of layers and their plane geometry:

```json
{                          
  "class": "gqa-windowed", 
  "from": 1,               
  "until": 60,             
  "step": 6,               
  "except": [11],          
  "kv_heads": 16,          
  "head_dim": 512,         
  "dtype": "float16",      
  "window_tokens": 1024,   
  "roles": ["key", "value"]
}                          
```

Layers covered: `from..until` stepped by `step`, minus `except`.
Validation is fail-closed: `from < until`; `until <= geometry.num_layers`
(bounded **before** any layer materialization, so a hostile `until` cannot
force a large allocation); `step >= 1`; `except` inside the range and unique;
the resulting layer set non-empty; `kv_heads`/`head_dim` > 0; `dtype`
currently must be `"float16"` (other values reserved); `roles` exactly
`["key", "value"]` at the current `kvpack-handoff` HEAD (the pair cursor has
not generalized; see the `mla-latent` note in 6.6 and drift D6); no layer in
two classes; and full **coverage** — the deduped union of class layers must
equal `0..geometry.num_layers` exactly (a holey table authenticates a cache
missing layers and is rejected). Per-class frame bound: for every class, the
per-plane byte count `min(window_tokens || cached, cached) × kv_heads ×
head_dim × 2` must fit the receiver's configured `max_frame_bytes` — checked
per class at arm, not as a cross-class average, so a mixed-class layout whose
largest class exceeds the cap fails at arm instead of mid-stream.
Flat-geometry agreement: with exactly one class,
`geometry.num_kv_heads`/`head_dim` must equal the class's; with multiple
classes they must be 0 ("see table").

Declared totals must match the walk:

```
expected_layer_frames  = sum over classes of |class layers| x |class roles|
expected_payload_bytes = sum over classes of |class layers| x |class roles|
                         x window x kv_heads x head_dim x 2                
```

### 6.6 The `mla-latent` class

The `mla-latent` layout class (`crates/kvpack/src/mla.rs`) stores a multi-head
latent attention layer as the exact model state — the per-token record
`c_KV ‖ k_rope` — instead of a K/V pair. On the wire it is one class with:

- `kv_heads = 1`, `head_dim = latent_dim + rope_dim` (the packed record),
  `dtype = "float16"`, `window_tokens = 0` (full coverage);
- `roles: ["key"]` — each layer ships a single Key-role plane carrying the
  `tokens × (latent_dim + rope_dim)` fp16 record, so `expected_layer_frames`
  and `expected_payload_bytes` follow the same class formulas with
  `|roles| = 1`;
- one durable family state per layer named `attn.kv_latent` (raw codec,
  contiguous, `direct` token axis), in place of `attn.k`/`attn.v`.

The geometry is derived from GGUF metadata (`attention.kv_lora_rank`,
`attention.qk_rope_head_dim`, `block_count`); a GGUF without them is not an
MLA model and is refused. The latent-to-per-head expansion is described by a
control-plane `MlaExpansionDescriptor` (canonical JSON, `deny_unknown_fields`,
fail-closed): `schema_version` = 1, `w_kvb_sha256` and `rope_config_sha256`
(64 lowercase hex), `latent_dim` (1..=65,536), `rope_dim` (1..=1,024),
`num_heads` (1..=1,024), `head_dim` (1..=1,024), and `target_layout`
(`"naive-per-head"` expands `c_KV @ W_KVb` with the rope path passed through
unchanged; `"absorbed-mqa"` consumes the latent directly and refuses
expansion). The W_KVb weight itself is engine tensor data and never enters
the format; its SHA-256 binds the exact matrix the expansion was qualified
against. Note the `roles: ["key"]` exception only validates against the
handoff revision pinned by `kvpack-cli`; the current `kvpack-handoff` HEAD
requires exactly `["key", "value"]` (drift D6).

### 6.7 Emission order (the walk), the schedule, and the windowing rule

Planes are emitted in declared table order: classes in table order, layers
ascending within a class, roles in the class's declared order. Sequence
numbers are the 0-based walk index. The v1 order (single class, layer-
ascending, key-then-value) is exactly the single-class special case.

**Wire schedule (M8).** The BEGIN `schedule` field selects the class
traversal order; it is a closed set and an unknown value fails validation.
Absent or `"layer-order"` is the declared order exactly — the field is
omitted from the JSON when absent, so pre-schedule begins stay byte- and
hash-identical. `"decode-priority"` is a stable partition: windowed classes
(`window_tokens > 0`, the newest cuts) stream before full-history classes,
each group keeping its declared relative order; layers stay ascending within
a class and roles stay in declared order (K-then-V for K/V classes). The
derivation is deterministic from the layout table at both ends. A `schedule`
on a v1 begin (empty `layout_table`) is rejected: there is no walk to
schedule. The schedule is not yet in the prose spec (drift D7).

The per-plane token range uses `cached = cached_token_count`:

```
window  = window_tokens == 0 ? cached : min(window_tokens, cached)
range   = [cached - window, cached)                               
```

so a full class ships `[0, cached)` and a windowed class ships only the
trailing in-window tokens (or the whole prefix if shorter than the window).

### 6.8 LAYER headers (v1 fields, plus v2 additions)

`byte_length`, `layer`, `logical_token_end`, `logical_token_start`, `role`
(`"key"`/`"value"`), `schema_version`, `sequence`, `sha256` (lowercase hex of
the plane bytes), `shape` `[tokens, kv_heads, head_dim]`, `transfer_id`.
v2 adds `dtype` (optional; absent means `float16`, valid only for
`float16` classes) and `layout_class` (required when the BEGIN carries a
table; must match the class owning that walk position).

Per-plane validation (fail-closed): correct `(class, layer, role)` at this
sequence; `logical_token_end == cached`; `logical_token_start == cached -
min(window_tokens || cached, cached)`; `shape == [end - start, kv_heads,
head_dim]`; `byte_length == (end - start) x kv_heads x head_dim x 2`; valid
sha256 form (payload hash checked on receipt); in v1 both v2 fields absent.

### 6.9 Descriptor chain hash and artifact seal

```
descriptor_chain_sha256 =                                             
  SHA-256( "kvpack-live-descriptor-chain-v1\0" ||                     
           per header in walk order: canonical_json(header) || "\n" ) 
                                                                      
artifact_sha256 =                                                     
  SHA-256( "kvpack-live-artifact-v1\0" ||                             
           canonical_json(begin) || "\n" ||                           
           per header in walk order: canonical_json(header) || "\n" ||
           canonical_json(seal_core) )                                
```

The SEAL authenticates `frame_count`, `payload_bytes`, `payload_sha256`, the
descriptor chain hash, the token-IDs digest (recomputed from the embedded
`prompt_token_ids`, whose length must be `cached + 1`), the deadline window,
and recomputes the artifact seal over everything above — so the layout table
and every header in walk order are bound into the terminal digest. ACK echoes
`artifact_sha256` with status `"committed"`; ABORT carries a `code`.

### 6.10 Worked example

A 3-layer model, `cached = 10` tokens, two classes:

```json
"layout_table": [                                                   
  { "class": "gqa-full",     "from": 0, "until": 3, "step": 2,      
    "except": [], "kv_heads": 8, "head_dim": 64, "dtype": "float16",
    "window_tokens": 0,    "roles": ["key", "value"] },             
  { "class": "gqa-windowed", "from": 1, "until": 2, "step": 1,      
    "except": [], "kv_heads": 8, "head_dim": 64, "dtype": "float16",
    "window_tokens": 4,    "roles": ["key", "value"] }              
]                                                                   
```

Class `gqa-full` covers layers {0, 2}; class `gqa-windowed` covers layer {1}.
Per-plane bytes: full `10 x 8 x 64 x 2 = 10,240`; windowed
`min(4, 10) x 8 x 64 x 2 = 4,096`. The walk:

```
+------+---------------+-------+-------+-----------+-----------------+
| seq  | class         | layer | role  | tokens    | payload bytes   |
+------+---------------+-------+-------+-----------+-----------------+
| 0    | gqa-full      | 0     | key   | [0, 10)   | 10,240          |
| 1    | gqa-full      | 0     | value | [0, 10)   | 10,240          |
| 2    | gqa-full      | 2     | key   | [0, 10)   | 10,240          |
| 3    | gqa-full      | 2     | value | [0, 10)   | 10,240          |
| 4    | gqa-windowed  | 1     | key   | [6, 10)   | 4,096           |
| 5    | gqa-windowed  | 1     | value | [6, 10)   | 4,096           |
+------+---------------+-------+-------+-----------+-----------------+
```

so `expected_layer_frames = 6` and `expected_payload_bytes = 49,152`; because
the table has two classes, `geometry.num_kv_heads` and `geometry.head_dim`
must be 0, and `geometry.num_layers` must be 3. On the durable side, the
consumer's descriptor registry admits the layout by name and derives one
family state per (class, layer, role): `gqa-windowed` states carry the
`TailWindow` token-axis rule, contiguous layout, and segment shapes whose
token dimension is 4 rather than 10.

---

## 7. Publication and crash-safety model

Durability is a two-phase protocol: bytes first, authority second. Object
bytes are staged, fsynced, and published **no-replace**; only after the bytes
are durable does a single catalog transaction make them visible. The SQLite
catalog (WAL, `synchronous=FULL`, one writer) is the sole authority on what
exists.

Object publication (per chunk/pack file):

1. Create a randomly named `*.partial` file in the staging directory
   (`O_CREAT|O_EXCL`, mode `0600`).
2. Write the complete object bytes; `fsync` the file.
3. Publish by **no-replace hard link** from the partial name to the final
   content-addressed path. If the target already exists it must be
   byte-identical (verified in full), otherwise the conflict is quarantined —
   bytes are never overwritten in place.
4. Unlink the partial name; `fsync` the target directory and the staging
   directory.

Catalog commit (single transaction): insert manifest and chunk rows, bump
chunk refcounts, write location rows (`AVAILABLE`), upsert reusable prefix
checkpoints, settle the byte reservation against the actual durable bytes
(fail if exceeded), and flip the upload state machine
`INIT -> RESERVED -> RECEIVING -> VERIFIED -> PUBLISHED`. Committing before
this point is impossible: the final `PUBLISHED` update requires the `VERIFIED`
state in the same transaction.

```
+---------------+----------------------------+---------------------------+
| writer        | object filesystem          | catalog (SQLite WAL)      |
+---------------+----------------------------+---------------------------+
| stage bytes   | create *.partial (EXCL)    |                           |
|               | write_all                  |                           |
|               | fsync(file)                |                           |
| publish       | link(partial, final)       |                           |
| (no replace)  | unlink(partial)            |                           |
|               | fsync(target dir)          |                           |
|               | fsync(staging dir)         |                           |
| commit        |                            | BEGIN IMMEDIATE           |
|               |                            | insert manifests/chunks   |
|               |                            | refcounts, locations,     |
|               |                            | prefix checkpoints        |
|               |                            | upload -> PUBLISHED       |
|               |                            | COMMIT (WAL, sync=FULL)   |
+---------------+----------------------------+---------------------------+
```

Crash behavior at each labelled point:

```
+---+-----------------------------------+--------------------------------+
| # | crash point                       | recovery behavior              |
+---+-----------------------------------+--------------------------------+
| A | before/at partial fsync           | partial may be torn or absent; |
|   |                                   | swept on restart; catalog      |
|   |                                   | never saw it                   |
| B | after fsync, before no-replace    | durable orphan partial; swept; |
|   | link                              | catalog never saw it           |
| C | after link, before dir fsyncs     | final name may or may not      |
|   |                                   | survive; unreferenced objects  |
|   |                                   | reconciled/GC'd; idempotent    |
|   |                                   | retry re-verifies bytes        |
| D | after dir fsyncs, before catalog  | object durable but invisible;  |
|   | commit                            | orphan reconciled by recovery; |
|   |                                   | upload re-driven by idempotency|
| E | after catalog commit              | fully durable and visible;     |
|   |                                   | nothing to do                  |
+---+-----------------------------------+--------------------------------+
```

Additional rules:

- **Idempotent retry.** Publication is keyed by an idempotency identity; a
  retried publish finds either no target (republishes), an identical target
  (byte-verified, no-op), or a conflicting target (quarantined with an audit
  event, never silently replaced).
- **Recovery reconciliation.** On open, recovery sweeps staging directories,
  aborts uploads left in non-terminal states, and cross-checks catalog rows
  against on-disk objects (fsck re-verifies bounds and digests; conflicting
  duplicate objects are moved to a bounded quarantine area).
- **Shadow-buffer restore rule.** Restore never decodes into live engine
  state. Chunk objects are authenticated (stored digest, header digest, object
  key, AEAD, canonical re-encode, content ID) and decoded into shadow buffers
  sized from the leaf manifest's `complete_physical_bytes`; the whole parent
  chain is authenticated before use. On the strictest tier the shadow
  allocation must equal the manifest's restored bytes exactly; only a fully
  verified shadow is handed to the engine.

---

## 8. Rejection order: the fail-closed matrix

The order below is the implemented decode order (`pack.rs`, `chunk.rs`,
`canonical.rs`, `validator.rs`); the first matching row wins.

Pack file (`decode_authenticated_pack`):

```
+-------------------------------------+----------------+---------------+
| malformed input class               | first check    | error class   |
+-------------------------------------+----------------+---------------+
| file smaller than 8192 bytes        | outer size     | Truncated     |
| wrong/legacy magic (IOKVPK1, etc.)  | header magic   | BadMagic      |
| version, header size, alignment !=  | header contract| BadMagic      |
| unknown flag bit set                | flags mask     | Reserved      |
| nonzero reserved bytes/salt/nonce   | reserved scan  | Reserved      |
| manifest length over bound          | length bounds  | Bounds        |
| footer magic/version/reserved bad   | footer framing | BadMagic/     |
|                                     |                | Reserved      |
| footer/header linkage mismatch      | length/epoch/  | Bounds        |
|                                     | id equality    |               |
| header bytes corrupted              | header digest  | Checksum      |
| body/length/epoch/file-size forged  | footer HMAC    | Authentication|
| ciphertext tampered                 | AEAD verify    | Authentication|
| plaintext length or manifest ID bad | identity check | Authentication|
| non-canonical manifest encoding     | decode+re-enc. | Reserved      |
| manifest tenant/epoch != header     | header binding | Authentication|
| semantic rule (spans, groups, etc.) | validate_manifest| Semantics/  |
|                                     |                | Bounds/Graph  |
+-------------------------------------+----------------+---------------+
```

Chunk object (`decode_chunk`):

```
+-------------------------------------+----------------+---------------+
| malformed input class               | first check    | error class   |
+-------------------------------------+----------------+---------------+
| length unaligned / over cap / short | object length  | Bounds        |
| wrong magic / version / sizes       | header contract| BadMagic      |
| unknown flag bit                    | flags mask     | Reserved      |
| codec not in {raw, lossless}        | codec enum     | UnknownEnum   |
| codec version != 1                  | version check  | Codec         |
| size fields inconsistent            | size arithmetic| Bounds        |
| payload length vs AEAD mismatch     | payload math   | Bounds        |
| payload truncated                   | payload end    | Truncated     |
| nonzero padding / reserved / salt   | zero scans     | Reserved      |
| size fields vs manifest reference   | reference sizes| Bounds        |
| object bytes corrupted              | stored digest  | Authentication|
| header bytes corrupted              | header digest  | Authentication|
| codec != family codec               | family match   | Semantics     |
| tenant / family / epoch mismatch    | binding check  | Authentication|
| content ID / object key mismatch    | key recompute  | Authentication|
| ciphertext tampered                 | AEAD verify    | Authentication|
| frame magic != codec / bad framing  | codec frame    | BadMagic/     |
|                                     |                | Reserved      |
| non-canonical packetization         | decode+re-enc. | Reserved      |
| plaintext identity mismatch         | content ID     | Authentication|
+-------------------------------------+----------------+---------------+
```

Canonical objects (family, schema, manifest, state declaration): wrong magic,
version, or nonzero reserved field; count outside bounds; truncation; unknown
enum (fail-closed `from_wire`); trailing bytes; non-canonical re-encode — in
read order, all before `validate_manifest` semantics (zero identities,
ordering, dependency graph, shape/stride/footprint checks, span partitioning,
atomic-group partition, totals).

Live wire (`frame.rs`, `manifest.rs`): bad magic/version/reserved; unknown
kind; JSON length 0 or > 1 MiB; payload over 64 MiB or nonzero on a non-LAYER
frame; LAYER `byte_length != payload_len`; non-canonical JSON (including
unknown fields); BEGIN contract (protocol/ABI/strategy, geometry bounds,
layout-table validation — per-class frame bound, coverage, roles, schedule —
declared totals, precision set, identity labels, deadline) — all before any
payload allocation; then per-plane walk position, range, shape, byte count,
hashes; then SEAL chain and artifact recomputation.

---

## 9. Qualification status

- **Everything here is draft and pre-freeze.** No byte in this document is a
  compatibility promise until the freeze gate ("Z1"). The prose specs say the
  same for themselves.
- **Durable objects (PACK_V1, CHUNK_V1, MANIFEST_V1, IDENTITY_V1).**
  Implemented in `kvpack-core` with a C reference decoder and a Python
  reference; error strings are pinned against the Python reference by
  conformance tests. Qualified lanes: raw and lossless codecs, plaintext and
  ChaCha20-Poly1305 objects, full and delta (depth 1..=7) manifests. Chunk
  headers may carry the optional M7 statistics sidecar (4.3); sidecar-absent
  objects are byte-identical to the pre-sidecar format, and pre-sidecar
  readers reject sidecar objects as non-canonical reserved bytes.
- **Closed codec set.** raw = 1 and lossless = 2 only. Q8 has **no production
  frame** (its draft magics are unknown values); Q4/Q2 and zlib frames are
  rejected as unknown enum/magic. Adding a codec is a format revision, not a
  configuration.
- **TailWindow / windowed classes.** `TokenAxisRule::TailWindow = 3` exists in
  the code and portable mode admits contiguous TailWindow states; the live v2
  protocol ships windowed classes and the persist path durably publishes them
  (see drift D2/D3 — the prose specs lag the code here).
- **Live handoff.** v1 (`canonical-kv-f16-le-v1`) is the pinned baseline; v2
  (`canonical-kv-v2`) is additive over it: empty `layout_table` means v1
  semantics byte- and hash-identically, and an absent `schedule` means
  `layer-order` byte- and hash-identically (6.7). The v2.1 `state-blob` plane
  kind (recurrent/conv state for hybrid models) is reserved and must not be
  used.
- **Red-team findings now closed in code and spec.** The 2026-08-02 pre-beta
  review (`docs/RED_TEAM_2026-08-02.md`) flagged that the BEGIN per-frame
  bound checked the *average* plane size rather than the per-class maximum,
  and that a malformed `until` could force a large allocation before
  validation. Both are fixed in the current `kvpack-handoff`: the per-class
  maximum frame size is checked at arm, and `until <= geometry.num_layers`
  is proven before any layer materialization (6.5); both rules are now in
  `spec/KVPACK_LIVE_V2.md` section 2, along with the full-coverage
  requirement the code enforces.

---

## 10. Spec/code drift register

Where the prose specs and the reference implementation disagree, this document
follows the code. Known drift (status as of this revision):

```
+----+----------+---------------------------------+---------------------------+
| ID | status   | spec says                       | code does                 |
+----+----------+---------------------------------+---------------------------+
| D1 | RESOLVED | KVPACK_LIVE_V2 section 7 now    | handoff manifest          |
|    |          | lists the closed weights set as | validation accepts        |
|    |          | {q4_k_m, nvfp4, mxfp4, bf16} —  | exactly those four        |
|    |          | matching the code               | labels (bf16 = the        |
|    |          |                                 | byte-matched BF16 lane);  |
|    |          |                                 | the label is still        |
|    |          |                                 | exact-matched, so         |
|    |          |                                 | unknown labels fail       |
| D2 | RESOLVED | MANIFEST_V1 now lists           | TokenAxisRule::           |
|    |          | TailWindow = 3 and the exact    | TailWindow = 3; portable  |
|    |          | portable full-snapshot range    | full snapshots admit the  |
|    |          | rules.                          | compact trailing range.   |
| D3 | OPEN     | MANIFEST_V1: realized-state     | validator rejects any     |
|    |          | `window` is "exactly zero for   | nonzero `window` today;   |
|    |          | ordinary causal KV", implying   | windowing is expressed    |
|    |          | nonzero windows are             | via TailWindow + segment  |
|    |          | representable                   | shapes                    |
| D4 | OPEN     | KVPACK_LIVE_V2 section 1        | frame.rs defines a        |
|    |          | describes the envelope only as  | 20-byte header with       |
|    |          | "unchanged from v1" without a   | BIG-ENDIAN lengths; the   |
|    |          | byte layout                     | layout exists only in     |
|    |          |                                 | code (section 6.1 here)   |
| D5 | STANDING | prefill v2 identities were      | prefill/v2.rs derives     |
|    | (break)  | derived under the               | engine-cache-abi/v3       |
|    |          | engine-cache-abi/v2 domain over | (5.10), which binds the   |
|    |          | bare numeric class geometry     | class label and state-    |
|    |          | (same-geometry layouts of       | name derivation into the  |
|    |          | different classes collided)     | identity. DELIBERATE      |
|    |          |                                 | BREAK: all v2-derived     |
|    |          |                                 | engine_cache_abi values   |
|    |          |                                 | change; v2 is retained    |
|    |          |                                 | for the existing          |
|    |          |                                 | qualified lanes as a      |
|    |          |                                 | test fixture              |
| D6 | OPEN     | KVPACK_LIVE_V2 section 2: class | mla-latent classes ship   |
|    |          | `roles` MUST be exactly         | roles: ["key"] (6.6).     |
|    |          | ["key", "value"], and current   | They validate only        |
|    |          | kvpack-handoff HEAD enforces    | against the handoff rev   |
|    |          | exactly that                    | pinned by kvpack-cli      |
|    |          |                                 | (d3501cd: roles non-empty |
|    |          |                                 | and unique). Cross-repo   |
|    |          |                                 | rev skew                  |
| D7 | OPEN     | KVPACK_LIVE_V2 section 2 lists  | manifest.rs adds a third: |
|    |          | two v2 BEGIN additions          | optional `schedule`       |
|    |          | (layout_table, portable_abi)    | (layer-order default      |
|    |          |                                 | absent, decode-priority;  |
|    |          |                                 | 6.7). Code-only for now   |
+----+----------+---------------------------------+---------------------------+
```

D1 was recorded in the red-team review (`docs/RED_TEAM_2026-08-02.md`, finding
7) and is now closed: the spec text and the code agree. D2/D3 remain open —
`spec/MANIFEST_V1.md` still lags the code — and `docs/SUPPORT_MATRIX.md`
already reflects the code position (sliding/rotating-window K/V is accepted
via `TailWindow` windowed classes in the v2 lane), consistent with section 9.
D4 remains open. D5 is the WS-C identity break, standing by design. D6 and D7
are new entries from the M8/MLA waves; both are pre-freeze and expected to
close when the spec and the pinned handoff rev catch up.

---

*Normative references: `spec/PACK_V1.md`, `spec/CHUNK_V1.md`,
`spec/MANIFEST_V1.md`, `spec/IDENTITY_V1.md`, `spec/KVPACK_LIVE_V2.md`;
reference implementations `crates/kvpack-core` (`pack.rs`, `chunk.rs`,
`manifest.rs`, `identity.rs`, `ids.rs`, `keys.rs`, `canonical.rs`, `enums.rs`,
`validator.rs`, `consts.rs`, `stats.rs`, `transform.rs`, `rotation.rs`), `crates/kvpack`
(`store_key.rs`, `store/keys.rs`, `store/publication/`, `store/recovery.rs`,
`restore/`, `mla.rs`, `prefill/v2.rs`),
and `crates/kvpack-handoff` (`frame.rs`, `manifest.rs`, `canonical.rs`).
Until the freeze, the code is the final arbiter of every byte described
here.*
