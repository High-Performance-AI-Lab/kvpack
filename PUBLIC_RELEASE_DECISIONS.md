# Initial public release decisions

This log records the source and runtime boundary for the first public
candidate. It does not authorize visibility changes, publishing, tags,
signing, or a release.

| Surface | Decision |
| --- | --- |
| Durable cache and exact replay | Public pre-release foundation |
| Rotation ABI | Public identity reservation; no consumer promotion claim |
| Spark/disaggregated prefill | Evidence-only PoC; not a production route |
| `kvpack-handoff` | Exact HTTPS Git dependency; never represented as vendored |
| Model-backed tests | `KVPACK_QWEN25_MODEL` / `KVPACK_GEMMA4_MODEL`; skip when absent |
| Lab hosts, keys, binaries, models and results | Explicit caller inputs; excluded from public snapshots |
| Research and qualification apparatus | Excluded from the initial fresh-history snapshot |
| Package version | Remains `0.1.0-alpha.1`; owner controls publication |

History contains development-only paths and lab apparatus. It will not be
rewritten. Public candidates are deterministic, allowlisted, fresh-history
snapshots tied to an exact private candidate commit.
