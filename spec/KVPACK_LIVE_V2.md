# kvpack live handoff — protocol v2 (canonical-kv-v2)

Status: pinned wire contract, additive over v1. v1
(`canonical-kv-f16-le-v1`) remains valid unchanged: a v1 begin (empty
`layout_table`) and v1 plane headers (absent `dtype`/`layout_class`)
parse and validate exactly as before, byte- and hash-identical. The v1
geometry tuples are subsumed as single-class layouts.

Normative references: `docs/PROTOCOL_V2_DESIGN.md` (rationale and
closed-assumption catalog), `crates/kvpack-handoff/src/manifest.rs`
(validation), `scripts/gx10/protocol.py` (producer walk),
`crates/kvpack/src/prefill.rs` (descriptor registry).

## 1. Framing

Unchanged from v1: `KVHF` frame envelope (magic, schema version, kind,
JSON manifest, optional payload for layer frames), canonical JSON
(sorted keys, ASCII, no floats), per-frame length bounds. Frame kinds:
BEGIN, LAYER, SEAL, ABORT, ACK.

## 2. BEGIN (v2)

All v1 fields unchanged and required. Two additions:

- `layout_table`: array of layer classes (MAY be empty = v1 semantics).
- `portable_abi`: MUST be `canonical-kv-v2` when `layout_table` is
  non-empty; MUST be `canonical-kv-f16-le-v1` when empty.

Each layout class:

```json
{
  "class": "gqa-full",            // registered label (ASCII, <=64)
  "from": 0,                      // first layer (inclusive)
  "until": 60,                    // last layer (exclusive)
  "step": 1,                      // layer stride (>= 1)
  "except": [5, 11],              // layers excluded from the range
  "kv_heads": 16,
  "head_dim": 512,
  "dtype": "float16",
  "window_tokens": 0,             // 0 = full prefix; >0 = trailing window
  "roles": ["key", "value"]       // declared emission order
}
```

Validation (fail-closed):

- Classes: `from < until`, `until <= geometry.num_layers` (bounded BEFORE
  any layer materialization, so a hostile `until` cannot force a large
  allocation), `step >= 1`, `except` entries within `[from, until)` and
  unique, resulting layer set non-empty; `kv_heads`/`head_dim` > 0;
  `dtype` = `float16` (other values reserved); `roles` MUST be exactly
  `["key", "value"]` — any other order or arity is reserved until the
  pair cursor generalizes.
- Non-overlap: no layer may appear in two classes.
- Coverage: the deduped union of class layers MUST equal
  `0..geometry.num_layers` exactly. A holey table authenticates a cache
  missing layers and is rejected.
- Per-class frame bound: for every class, the per-plane byte count
  `min(window_tokens || cached, cached) × kv_heads × head_dim × 2` MUST
  fit the receiver's configured `max_frame_bytes`. The bound is per class
  (the maximum any single frame can be), not the cross-class average —
  mixed-class layouts whose largest class exceeds the cap fail at arm
  instead of mid-stream.
- Flat-geometry agreement: with exactly one class, `geometry.num_kv_heads`
  and `geometry.head_dim` MUST equal the class's; with multiple classes
  they MUST be 0 (meaning "see table").
- `expected_layer_frames` MUST equal the walk length: sum over classes
  of `|class layers| × |class roles|`.
- `expected_payload_bytes` MUST equal the walk's byte count: sum over
  classes of `min(window_tokens || cached, cached) × kv_heads ×
  head_dim × 2 × |class roles|`.

## 3. Emission order (the walk)

Planes are emitted in **declared table order**: classes in table order,
layers ascending within a class, roles in the class's declared order.
Sequence numbers are the walk index (0-based). The v1 order (single
class, layer-asc, key-then-value) is the single-class special case.

## 4. LAYER headers (v2)

All v1 fields unchanged and required. Two additions, both OPTIONAL but
required when the begin carries a layout table:

- `dtype`: plane dtype tag; MUST equal the owning class's `dtype`.
  Absent means `float16` and is valid only for `float16` classes.
- `layout_class`: the owning class's `class` label; MUST be present and
  exist in the begin's table.

Per-plane validation (fail-closed): class exists; `(layer, role)`
matches the walk position; `logical_token_end == cached`;
`logical_token_start == cached − min(window_tokens || cached, cached)`;
`shape == [end − start, class.kv_heads, class.head_dim]`;
`byte_length == (end − start) × kv_heads × head_dim × 2`; sha256
lowercase-hex.

## 5. SEAL / ACK / ABORT

Unchanged: descriptor-chain and artifact hashes bind the begin
(including the layout table) and every header in walk order; the seal
authenticates frame count, payload bytes, payload hash, chain hash,
token digest, and deadline. ABORT/ACK forms unchanged.

## 6. Descriptor registry (durable side)

`crates/kvpack/src/prefill.rs` `PORTABLE_PREFILL_LAYOUTS_V2` registers
admitted layouts by name (`qwen2.5-7b`, `gpt-oss-120b`, `gemma4-31b`).
Unregistered layout names hard-error. `derive_portable_prefill_descriptor_v2`
emits per-class states (`elements_per_token`, `TailWindow` token axis
for windowed classes, per-class byte bounds) in canonical
layer-ascending order, independent of the wire walk.

## 7. Closed weights set

`precision.weights ∈ {q4_k_m, nvfp4, mxfp4, bf16}`; `precision.compute`
and `precision.kv` remain `float16`. `q4_k_m` is the GGUF lane, `nvfp4`
the modelopt FP4 producer with deterministic F16-dequant consumer,
`mxfp4` the gpt-oss producer, and `bf16` a byte-matched BF16 GGUF on both
ends (the Gemma 4 31B lane). The consumer exact-matches the label
against its configured expectation, so an unexpected label stays
fail-closed.

## 8. Reserved (v2.1, not built)

`state-blob` plane kind for hybrid models (DeltaNet/Mamba recurrent
state, no token axis): header `{ kind, layer, state: "conv" |
"recurrent", dtype, shape }`; `CacheKind::RecurrentState` /
`ConvState` in kvpack-core enums. Reserved labels MUST NOT be used
until v2.1 is pinned.
