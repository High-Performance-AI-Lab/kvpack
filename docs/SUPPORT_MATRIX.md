# Closed v1 format support matrix

Status: pre-release format disposition. This matrix describes values the v1
format can represent; it does not claim that any production route or hardware
profile has been qualified. An item marked rejected has no accepted v1 wire
value, flag, or extension block.

## Durable semantics

| Dimension | Accepted v1 | Rejected in v1 | Rule |
| --- | --- | --- | --- |
| state family | ordinary causal-transformer K/V; sliding/rotating-window K/V via `TokenAxisRule::TailWindow` windowed classes in the evidence-only persist-prefill-v2 foundation | MLA, KDA, Mamba/SSM, convolution, terminal recurrent snapshots, hybrid state | rejected kinds are unrepresentable, not merely runtime-disabled |
| checkpoint form | standalone full cut; ordinary-KV append delta | decorative parents, per-state delta switches, partial-family delta | one full base plus at most seven complete deltas |
| cut | exact cut zero fallback, each admitted 256-token cut, exact partial final cut | ancestor row aliasing another cut | every nonzero catalog cut names an independently authenticated artifact |
| payload codec | raw; bounded lossless exact frame | Q8, Q4, Q2, any lossy codec | codec version and exact decoded bytes are identity material |
| encryption | plaintext authenticated object; qualified AEAD object | unauthenticated payload, caller nonce, implicit key epoch | content, ciphertext/object, stored digest, nonce, codec, and epoch identities remain distinct |
| representation mode | engine-native; canonical portable | inferred portability | portable byte integrity never claims numerical compatibility |
| continuation | none | RNG, sampler, grammar, tool, logits, resume, speculative metadata | requires a separately versioned post-v1 object if later authorized |
| auxiliary input | typed length-delimited opaque identity root | raw prompts, token witnesses, workload content | no raw or encrypted witness is retained |
| extensibility | exact versioned object only | unknown enum/flag/version, nonzero reserved byte, optional extension block | v1 is closed and fails unknown input before semantic use |

## Representation-family dimensions

All accepted values below are enumerated by immutable store/service policy; a
request cannot widen the configured set.

| Dimension | Accepted v1 rule |
| --- | --- |
| engine/cache ABI | exact opaque versioned identity |
| state inventory | ordered complete ordinary-KV state-key inventory |
| dtype | exact lossless dtype declared by qualified family; no implicit cast |
| layout | explicit contiguous or explicitly described strided native layout; canonical portable layout is separately identified |
| token axis | explicit per-state logical token axis and physical slicing rule |
| page size | nonzero fixed family value; baseline durable cut is 256 tokens |
| static dimensions | layer/head/head-dimension and other non-cut dimensions bind the family |
| topology | exact TP/PP/shard map and ordering; no implicit reshard |
| concrete shape/span | excluded from family; binds `RealizedCutSchemaId` for one cut |

## Platforms, key providers, and transport

| Surface | Production eligible | Probe/development only | Rejected behavior |
| --- | --- | --- | --- |
| local store | qualified macOS and Linux filesystems with SQLite/WAL | other Unix platforms | path-derived authority |
| key provider | macOS Keychain adapter; qualified Linux OS-key-store adapter | in-memory and file adapters; future Vault/cloud adapters behind same trait | embedding provider metadata or root keys in durable objects |
| agent transport | bounded AF_UNIX `SOCK_STREAM` with framed control and `SCM_RIGHTS` | none | macOS `SOCK_SEQPACKET`, paths in messages, writable/unbounded FDs |
| gateway transport | rustls TLS 1.3 mTLS, ALPN `h2`, real TCP; shared immutable-storage reads where qualified | loopback fixtures | cleartext, HTTP/1 fallback, identity inferred from request fields |
| remote data profile | bounded streaming TCP/shared storage | explicit RDMA, Thunderbolt RDMA, and GPUDirect probes | production routing from configuration, CPU fixture, or unqualified profile |
| Linux behavior | release-tested on Linux CI/dedicated runner | cross-compilation alone | claiming Linux qualification from macOS tests |

## Promotion policy

No engine or two-host route is promoted in the initial public release. Any
future promotion must bind raw or lossless exact bytes to an immutable workload,
candidate, apparatus, token-equivalence gate, and retained evidence. That work
does not add model-specific material to the durable wire.
