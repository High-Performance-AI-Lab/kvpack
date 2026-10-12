---
name: kvpack-engine-integration
description: Integrate kvpack's authenticated export, exact-prefix lookup, transactional restore, or live handoff into an inference engine such as llama.cpp or DwarfStar. Use when mapping engine KV or recurrent state into kvpack identities and state streams, implementing an engine restore sink, deciding whether an existing checkpoint format fits production-v1, or qualifying same-engine and cross-engine replay.
---

# Integrate an inference engine with kvpack

Work from the kvpack repository root. Read `AGENTS.md` when present, then inspect
the target engine at the exact revision being changed. Treat the current Rust
exports in `crates/kvpack/src/lib.rs` and the normative files in `spec/` as the
authority; do not revive names from old documentation without finding them in
the current workspace.

Read [the integration contract](references/integration-contract.md) for every
integration. Also read the target-specific guide:

- [llama.cpp](references/llama-cpp.md)
- [DwarfStar](references/dwarfstar.md)

For another engine, apply the shared contract and create a target-specific
state inventory before writing adapter code.

## Choose the claim before the representation

Classify the work as exactly one of these lanes:

1. **Native durable replay**: producer and consumer use the same engine cache
   ABI. Prefer this as the first integration.
2. **Native live handoff**: same representation, transferred through
   `kvpack-handoff` inside a separately authenticated transport.
3. **Portable cross-engine handoff**: different engines deliberately emit and
   consume one shared canonical representation. This requires its own ABI,
   transforms, and cross-engine exactness qualification.

Never turn a native checkpoint into a portable claim by renaming it. A cache
hit between two builds is allowed only when every semantic and representation
identity component agrees.

## Follow this implementation order

1. **Inventory the state.** Record every tensor or recurrent frontier needed
   after a cut, its logical token positions, dtype, shape, strides, layout,
   sharding, and synchronization boundary. Include logits or sampler state only
   if continuation semantics require them.
2. **Run the representation-fit gate.** Production-v1 state streams are
   fixed-width and token-indexed. If the engine only exposes a composite,
   variable-size snapshot, do not disguise it as a tensor plane. Add an
   explicit versioned artifact surface or expose canonical logical planes.
3. **Define identities.** Bind weights and config, adapters, tokenizer and chat
   template, position semantics, qualified math, engine cache ABI, topology,
   shard map, and the complete canonical state inventory. Change the ABI when
   bytes or restore meaning change.
4. **Implement export.** Quiesce device work, declare exact token IDs and every
   state before reading any source, stream states in canonical order through
   `ExportSession`, and publish only with `commit`.
5. **Implement restore.** Ask `LocalStore::restore_candidates` for candidates,
   build an `AuthenticatedRestorePlan`, and restore only through a
   `VerifiedRestoreSink` whose writes remain invisible until commit.
6. **Resume the suffix.** Continue from `matched_cut().token_count`; never
   silently treat an ancestor hit as an exact hit. Recompute is an explicit,
   healthy fallback.
7. **Qualify and fault-inject.** Compare uninterrupted execution with
   restore-and-continue, then test truncation, corruption, cancellation,
   allocation failure, commit failure, restart, and incompatible identities.

## Enforce the non-negotiable boundaries

- Exact token IDs are the prefix identity. Prompt text or semantic similarity
  is not a substitute.
- Never write verified chunks into live engine state before the complete plan
  is validated and staged.
- `abort_restore` discards every shadow allocation. `reset_restore` returns the
  live engine to a known-empty state after a failed commit attempt.
- Do not omit recurrent, sliding-window, compressor, indexer, speculative, or
  distributed-shard state merely because an ordinary attention model only has
  K and V planes.
- Do not reuse a representation family after changing layout, dtype, RoPE
  convention, cache shift/ring semantics, quantization, layer ownership, or
  restoration behavior.
- `kvpack-handoff` supplies framing and verification, not peer discovery,
  authentication, scheduling, or engine installation.
- Keep raw prompts, model paths, hostnames, keys, and private topology out of
  identities, telemetry, fixtures, and public receipts.

## Require completion evidence

Do not call an integration complete without:

- a checked-in state inventory and identity map;
- a restore sink with abort and failed-commit reset tests;
- exact miss tests for every incompatible identity field;
- uninterrupted-versus-restored continuation tests at multiple cut depths;
- cold-process export/restore coverage, not only an in-process round trip;
- fault injection proving no partial state becomes live;
- the workspace tests and deterministic wire-vector check passing.

Run at minimum:

```sh
cargo fmt --all --check
cargo test --workspace --all-targets --locked
scripts/check_wire_vectors.sh
```
