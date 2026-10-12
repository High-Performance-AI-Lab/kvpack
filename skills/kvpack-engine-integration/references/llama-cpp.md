# llama.cpp integration guide

## Verify the target revision first

The public llama.cpp state API evolves. Before editing, record the target
commit and inspect `include/llama.h`, `src/llama-context.cpp`, and the active
memory implementation. The baseline inspected for this guide was upstream
commit `8918deaa8ea79ad859dd73ab66f4c452fa70c4ce`.

At that revision the public per-sequence functions are:

```c
llama_state_seq_get_size(ctx, seq_id)
llama_state_seq_get_data(ctx, dst, size, seq_id)
llama_state_seq_set_data(ctx, src, size, dest_seq_id)
```

The `_ext` variants accept state flags. The public get/set entry points
synchronize the context before touching state. `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`
keeps tensor data in context-owned device storage and is not a durable,
cross-process byte representation; do not use it for a kvpack store artifact.

## Use the serializer as an oracle, not a tensor declaration

The sequence-state blob begins with private framing and delegates to the active
memory implementation. Its contents and size vary across ordinary KV,
sliding-window, recurrent, hybrid, and newer model-specific memories. It does
not include the request's authoritative token-ID sequence.

Therefore:

- use `llama_state_seq_get_data`/`set_data` to prove same-build round-trip
  behavior and as a baseline for fault tests;
- do not feed that composite blob directly to `ExportSession` unless a new
  first-class opaque artifact contract has been added and specified;
- keep exact token IDs in the kvpack request/declaration regardless of what a
  session file contains.

## Recommended first production integration

Start with ordinary attention models and expose canonical, logical cache rows
from the active llama memory implementation:

1. Add a narrow adapter boundary inside llama.cpp that can enumerate every
   state plane, synchronize the context, and stream logical rows without
   exposing raw allocator pointers to the kvpack side.
2. Describe K and V separately per layer using canonical `StateKey` order.
   Record dtype, token axis, fixed head dimensions, strides, cache positions,
   and whether RoPE has already been applied.
3. For a ring or shifted cache, export logical token order, not physical cell
   order. Use a native gathered layout or a qualified contiguous normalized
   representation. Use `TailWindow` only when the engine can resume from the
   retained tail under the declared absolute-position semantics.
4. Restore into a new `llama_context` or an inactive cache generation. Only
   swap it into the serving session after every verified plane is installed.
5. Resume at the authenticated matched cut and evaluate only the suffix.

Do not assume every llama model is a K/V-only model. Inventory the active
`llama_memory_i` implementation. SWA, recurrent/Mamba, hybrid memory, DSA/MSA,
DSV4 compressed state, and MTP paths can require additional rows, counters, or
frontiers. Refuse unsupported memory kinds before export.

## Identity requirements

At minimum bind:

- model GGUF content/config identity and ordered adapters;
- tokenizer assets, special tokens, and the exact chat template;
- llama.cpp cache ABI version plus a pinned source/build identity;
- context parameters that change cache allocation or interpretation;
- K/V types, flash-attention choice, offload/backend mode, RoPE/scaling,
  sliding-window/cache-shift rules, and sequence semantics;
- tensor split, layer placement, and shard ownership;
- the numerical lane qualified for continuation.

A private struct layout, compiler padding, pointer value, or mutable branch
name is not a stable ABI identity.

## Transactional sink pattern

The safest first sink owns a fresh context:

1. `begin_restore` creates an unloaded context with the authenticated runtime
   configuration and allocates staging buffers.
2. `write_verified_chunk` fills only staging planes and checks state bounds.
3. `commit_restore` imports all planes into the fresh context, synchronizes,
   validates position/cell metadata, then swaps the session's context pointer
   while requests are excluded.
4. `abort_restore` destroys the fresh context.
5. `reset_restore` destroys any context involved in an uncertain swap and
   creates or selects a known-empty session.

Calling `llama_state_seq_set_data` on the live serving context is acceptable as
a local experiment, not as the production transaction boundary, because a
failed load may already have mutated memory.

## llama.cpp qualification

For each supported memory kind and backend:

- compare the public sequence-state round trip with the structured adapter;
- compare next-token logits before and after restore;
- compare at least a deterministic multi-token continuation;
- test multiple sequence IDs and reject state from the wrong sequence;
- test cache wrap, shift/defrag, SWA boundaries, and context-size mismatch;
- test CPU and each advertised accelerator backend separately;
- prove every unsupported model memory fails closed and recomputes.

Use the upstream source as the API authority:
<https://github.com/ggml-org/llama.cpp>.
