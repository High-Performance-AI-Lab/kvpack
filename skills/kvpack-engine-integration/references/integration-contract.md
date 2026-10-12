# kvpack engine integration contract

## Start from the shipping surface

The current production-v1 path is exported from `crates/kvpack/src/lib.rs`:

- storage: `LocalStore`, `StoreConfig`, and `StoreKey`;
- export: `ExportDeclaration`, `ExportStateDeclaration`, `ExportCutPolicy`, and
  `ExportSession`;
- lookup: `RestoreRequest` and `LocalStore::restore_candidates`;
- authenticated planning: `AuthenticatedRestorePlan` and `RestoreLimits`;
- installation: `VerifiedRestoreSink`, `RestoreCancellation`, and
  `InstalledRestore`;
- optional pre-staging: `AuthenticatedRestorePlan::prestage_shadow` and
  `ShadowRestoreHandle::promote_if`;
- live transfer: the separate `kvpack-handoff` crate.

Use these tests as executable examples:

- `crates/kvpack/tests/cut_export/`
- `crates/kvpack/tests/restore_plan/`
- `crates/kvpack/tests/production_v1/`
- `crates/kvpack/tests/fidelity/main.rs`
- `crates/kvpack-handoff/tests/live_bundle/`

Do not base a new adapter on a symbol unless `rg` finds it in the current
workspace. In particular, historical adapter or bridge names are not the
shipping production-v1 integration surface.

## Representation-fit gate

Every `RepresentationFamilyId` state has exactly one token dimension and a
fixed number of bytes per logical token. `ExportSession` therefore expects
exactly:

```text
plaintext bytes = input token count × state bytes per token
```

`TokenAxisRule::TailWindow` handles a trailing logical window, and native
layouts may use declared strides or gathers, but the state still needs a
well-defined token axis. This works naturally for canonical K/V rows and other
fixed-width token-indexed planes.

A native serializer often emits a different object: headers, variable-length
cell tables, tokens, logits, counters, recurrent frontiers, and several tensor
classes in one blob. That blob is valuable as a round-trip oracle, but it is
not automatically a valid production-v1 state stream.

If a target exposes only such a blob, choose one explicit design:

1. expose canonical logical state planes from the engine;
2. add and specify a first-class opaque exact-cut artifact surface in kvpack;
3. add a richer versioned state schema for non-token-linear state.

Do not pad, truncate, or relabel a composite snapshot to make validation pass.
Any new artifact/schema surface needs normative specification changes,
deterministic vectors, bounds, corruption tests, and an ABI identity.

## Identity map

Build `SemanticModelId` from stable digests of:

- `weights_config`: model weights plus architecture/configuration that changes
  inference state;
- `adapters`: the ordered adapter set and scales, or a canonical empty value;
- `tokenizer_template`: tokenizer assets, special-token mapping, normalization,
  and rendered chat template;
- `position_semantics`: RoPE/scaling convention, position IDs, sliding-window
  and cache-shift policy;
- `qualified_math`: the exact numerical lane that has passed continuation
  qualification.

Build `RepresentationFamilyId` from:

- `engine_cache_abi`: versioned byte layout and restore semantics;
- `mode`: `Native` unless two engines intentionally share the representation;
- `page_size_tokens`: checkpoint/chunk policy identity;
- `topology`: tensor/pipeline ownership and device-visible organization;
- `shard_map`: exact layer or tensor ownership;
- `states`: a complete, canonical `StateKey`-ordered inventory.

Hash canonical records, not absolute paths, mutable tags, pointers, struct
padding, hostnames, or display strings. A changed identity must produce a clean
miss, never a best-effort restore.

## Export lifecycle

At an engine-defined valid cut:

1. prevent new mutation of the session and synchronize accelerator work;
2. copy the exact `u32` token IDs and build the complete family/declaration;
3. call `ExportSession::begin` with `ExportCutPolicy::production_v1()` and a
   qualified `WritePolicy`;
4. call `next_state` in canonical family order and write exactly the declared
   bytes with `write_source` or `write_all` plus `finish`;
5. call `commit` only after every declared state succeeds.

Dropping an unfinished state writer poisons the whole session. A short source,
an extra byte, a read error, or out-of-order state must fail the export. Do not
publish a prefix until every state in its atomic group is durable.

## Lookup and restore lifecycle

For the requested exact token sequence:

1. build `RestoreRequest` with the same semantic model, family, auxiliary
   identities, minimum key epoch, and a bounded candidate count;
2. select a candidate by tier and suffix cost; retain the explicit recompute
   candidate as the normal fallback;
3. build `AuthenticatedRestorePlan::build` with resource limits;
4. reject or regenerate a plan where `requires_guided_recompute()` is true;
5. run `restore_sequential` or a bounded `restore_parallel` into an
   engine-owned `VerifiedRestoreSink`;
6. resume from `plan.matched_cut().token_count` and evaluate the suffix;
7. settle `InstalledRestore` with `engine_free()` when the engine no longer
   needs the associated restore lifetime. Do not silently drop unsettled
   handles; uncertainty intentionally retains resources.

Use `prestage_shadow` when lookup/verification can overlap other work. Promote
only with the exact authenticated manifest ID. Abort an unused shadow handle so
its reservation and pins are released.

## Restore sink contract

`VerifiedRestoreSink` is a transaction boundary:

- `begin_restore`: allocate a fresh, engine-invisible staging generation for
  the declared states and bounds;
- `write_verified_chunk`: bounds-check `(state, logical_offset, length)` and
  write only into that generation;
- `commit_restore`: atomically make the complete generation active;
- `abort_restore`: discard all staging and leave the prior live generation
  unchanged;
- `reset_restore`: after a failed commit attempt, force a known-empty or known-
  good engine state.

If an engine cannot atomically exchange cache generations, restore into a fresh
context/session and swap the owner pointer under the engine's request lock.
In-place restore with a promise to repair later is not transactional.

## Qualification matrix

For every supported model/layout/backend combination, compare an uninterrupted
run with export, process restart, restore, and suffix continuation. Cover:

- cuts below, at, and above a page/checkpoint boundary;
- exact hit, ancestor hit plus suffix, and miss;
- ordinary KV, sliding/ring wrap, recurrent/hybrid state, and speculative state
  when the engine supports them;
- next-token logits when available, greedy token continuation, and a longer
  deterministic continuation;
- corrupted manifest/chunk, truncation, wrong key epoch, and every identity
  mismatch;
- cancellation before begin, during writes, and before commit;
- sink allocation/write/commit failures and successful retry afterward;
- cold producer and cold consumer processes.

Byte equality is required for an exact raw/lossless state lane. Behavioral
qualification is additionally required because unchanged bytes can still be
installed with the wrong strides, positions, or engine semantics.

## Live and cross-engine handoff

Use `kvpack-handoff` only after the durable/native adapter contract is clear.
The surrounding system must still provide authenticated peers, authorization,
replay protection, cancellation, discovery, scheduling, deadlines, and
producer recovery.

For cross-engine reuse, define one portable ABI with canonical dtype, layout,
RoPE stage, positions, sharding, and transform versions. Qualify every
producer-consumer pair independently. A same-engine native checkpoint is not a
portable representation merely because the handoff frame can carry its bytes.
