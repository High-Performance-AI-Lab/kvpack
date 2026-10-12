# Model-derived layouts

Status: evidence-only foundation. `crates/kvpack/src/gguf_layout.rs` derives v2
layout tables from GGUF metadata (sidecar JSON for what GGUF cannot express),
and `kvpackctl store persist-prefill-v2` resolves exactly one of
`--layout-name` / `--model-gguf` / `--layout-json`. No public production route
or live two-host claim is made. The registry in `crates/kvpack/src/prefill.rs`
is a test oracle:
derived qwen2.5-7b and gpt-oss-120b layouts match it field-by-field,
and a gemma4-31b sidecar reproduces it. GGUF-derived and registry-named
persists produce identical durable manifests.

## Goal

A new model must require **zero kvpack changes** — not a registry entry,
not a geometry tuple, not a release — unless it carries a genuinely new
state kind (MLA, linear/recurrent, conv state), in which case a format
decision is deliberate and versioned, never improvised mid-lane.

The v2 wire is already generic: `layout_table` classes
(`from/until/step/except`, `kv_heads`, `head_dim`, `dtype`,
`window_tokens`, `roles`) describe every full/SWA hybrid surveyed
(`docs/BIGGER_MODEL_SURVEY.md`). Only the *derivation* of the table is
hand-maintained. This design replaces the registry with a compiler.

## Part 1 — GGUF-derived layouts (default path)

The lane tooling (harness + producer connector, never kvpack-core) parses
the model's GGUF metadata and emits the v2 layout table at arm time:

| GGUF key | Layout field |
| --- | --- |
| `*.block_count` | class `from/until` span (all layers) |
| `*.attention.head_count_kv` | `kv_heads` |
| `*.attention.key_length` / `value_length` | `head_dim` (per class) |
| `*.attention.sliding_window` (+ pattern ratio when present) | `window_tokens`, `step` on the SWA class |
| `*.rope.*` | recorded into expectations; never reinterpreted |
| `general.architecture` | fail-closed allowlist of derivable archs (llama/qwen2/gemma-family shapes) |

Rules:

- **Fail closed.** A missing key, an unrecognized architecture, or a
  pattern the table cannot express aborts the arm with `layout
  underivable` — kvpack never guesses geometry.
- **Derivation lives in the harness.** kvpack-core keeps its closed,
  validated semantics; it consumes the layout table, it never parses
  GGUF. The descriptor chain and per-plane sha256 stay authoritative.
- The hand registry becomes a *test oracle*: derived tables for
  qwen2.5-7b, gpt-oss-120b, and gemma4-31b must byte-match the registry
  entries until the registry is deleted.

## Part 2 — sidecar descriptors (escape hatch)

What GGUF cannot express — explicit per-layer class lists (e.g.
Inkling's 35-entry `local_layer_ids`), nonstandard attention schemes —
goes in a small JSON sidecar next to the model artifact, hashed into
the run's expectations like every other identity input. This is data,
not plugin code: validation stays in the receiver. `LAYOUT_JSON` in the
llamacpp lane is the prototype of this path and becomes the sidecar
loader.

## Part 3 — future route promotion contract

Correctness across architectures is necessary but does not promote a route. A
future lane could only be called supported if interleaved replication hides
the network cost: KV
planes stream per layer during prefill, and the wire must stay ahead of
compute. For each architecture at a given context:

- **Wire budget** (derived, not measured-after-the-fact):
  `bytes = Σ_classes layers × min(window_tokens, ctx) × kv_heads × head_dim × dtype_size`.
  Examples: qwen2.5-7b @32k = 1.88 GB; gemma4-31b @32k = 2.9 GB
  windowed (9.4 GB naive); GLM-4.5-Air @128k = 23 GB.
- **Overlap check** (measured): the producer's per-layer phase dump
  (`producer-timing.json`, already recorded) must show layer arrivals
  interleaved with compute — `last_layer_arrival_s` within a small
  bound of prefill-compute end, and `first_layer_arrival_s` early
  enough that the consumer is importing during compute, not after.
  Any future headline metric must be defined by an immutable workload contract
  and backed by retained evidence from the exact candidate.
- **Admission rule**: a new architecture's family gate (8k) must record
  the derived wire budget and the measured arrival spread in the run
  record; its scale gate is only staged when the *projected* wire
  budget at target context still amortizes on the declared transport and passes
  an explicit resource guard.

The initial public release does not satisfy this route-promotion contract. It
ships layout derivation and validation as an evidence-only foundation.

## Part 4 — wild architectures

MLA, KDA/DeltaNet, Mamba/SSM, conv state: no `CacheKind`, no plane kind,
no derivation. The design response is the reserved v2.1 `state-blob`
plane kind (`docs/PROTOCOL_V2_DESIGN.md` Part 5) — a versioned format
decision, exactly the "totally new wild architecture" exception. The
derivation compiler must refuse these archs loudly at parse time so the
exception is visible in the run record, not discovered as a token
mismatch.
