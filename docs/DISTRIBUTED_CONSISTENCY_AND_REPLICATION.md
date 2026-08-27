# Distributed consistency and replication architecture

Status: architecture recommendation, not yet implemented, 2026-07-29.

This document defines the distributed-systems boundary shared by
`kvpack-production-v1`, `kvpack-agent`, `kvpack-gateway`, `kvpack-cache`, and
CacheWeaver. It is authoritative for the intended consistency model, but it
does not widen the production-v1 cache surface. The first qualified inference
lane remains Qwen2.5 ordinary K/V; Kimi, MLA/KDA, approximate reuse, and mobile
mesh operation are outside that release gate.

The central decision is not to make the whole system a CRDT. Immutable,
authenticated objects are safe to replicate without coordination. Mutable
facts that can only improve discovery or policy may converge asynchronously.
Any decision that can corrupt state, over-allocate a resource, authorize a
restore, revoke access, or delete live bytes retains one fenced authority.

## Governing rule

Use the following test for every distributed field and transition:

> If stale or conflicting information can only cause an extra miss, retry, or
> recomputation, it may be eventually consistent. If it can cause incorrect KV,
> unauthorized access, hard-quota violation, split-brain publication, premature
> reclamation, or partial engine installation, it must be owned and fenced.

This is the system-specific application of the CALM and invariant-confluence
results: monotone facts can be accumulated without a global order, while
non-monotone invariants may require coordination. CRDTs solve convergence; they
do not establish truth, authorization, exclusive ownership, revocation, or safe
garbage collection.

The existing architecture already supplies the necessary safety boundary:

- `.kvchunk` and `.kvpack` objects are immutable and authenticated;
- each local SQLite/WAL catalog is authoritative for its own bytes and lifecycle;
- `kvpack-cache` supplies candidates, placement, and policy, but never proves
  that bytes exist or authorize restore;
- every consumer authenticates selected bytes against exact expectations before
  engine-visible commit;
- global control-plane failure degrades to a verified local candidate or
  recomputation;
- CacheWeaver begins only after an exact miss and cannot mint exact provenance.

These invariants must survive every replication mechanism introduced below.

## Consistency matrix

| State | Owner and mechanism | Required consistency | Failure consequence |
|---|---|---|---|
| Chunk and manifest identity | `kvpack`; immutable content-addressed objects | Grow-only/set-union replication | Duplicate transfer; conflicting bytes under one ID are corruption |
| Local object existence | Local store or gateway SQLite/WAL | Single-owner transactional | Local miss or fail-closed readiness |
| Replica/location advertisements | `kvpack-cache`; dotted observed-remove set | AP/eventually consistent | Stale positive fails verification; stale negative recomputes |
| Prefix-residency hints | `kvpack-cache`; tenant-scoped summaries | AP/eventually consistent | Extra lookup or miss |
| Popularity, reuse, and cost observations | Per-origin time buckets/sketches | AP/eventually consistent | Temporarily suboptimal placement or eviction |
| Topology and load observations | Membership gossip plus reporter incarnation/sequence | AP/eventually consistent | Route away, retry, or replicate unnecessarily |
| CacheWeaver proposals and evidence | Immutable signed events/provenance DAG | AP/eventually consistent | Delayed learning or duplicated evaluation |
| Publication generation and idempotency | Catalog owning the upload | Linearizable compare-and-swap | Stale publisher is rejected |
| Active gateway writer | Small CP authority plus storage-checked fence | Linearizable term/lease | No split-brain writes |
| Admission grant and engine install | Resource-owning agent/engine | Single-owner transactional and fenced | No double allocation or partial install |
| Read/source lease and physical GC | Object owner | Ordered delete barrier plus lease/pin drain | Live bytes are never reclaimed |
| Key epoch, revocation, and minimum readable epoch | KMS/configuration authority | Strongly ordered and fail closed | Revoked data cannot be restored |
| Qualification/policy promotion | Release/policy authority | Strongly ordered signed pointer | Unqualified representations cannot become production routes |
| Hard quota and endurance budget | Central reservation or escrow rights | Invariant preserving | Aggregate use cannot exceed the configured bound |

The consistency designation applies to authority, not transport. All messages,
including CP messages, remain retryable and idempotent.

## Replicated event contract

Do not replicate SQLite files or merge WALs. Keep one writer for each local
state transition and place a replicated event layer around the authoritative
transaction.

Every externally relevant transaction writes its local state and an outbox row
atomically. A bounded background worker forwards the event at least once. Each
receiver records an inbox entry and applies the event idempotently in one local
transaction.

A versioned event envelope contains at least:

```text
ReplicationEventV1 {
    tenant_namespace
    origin_node_id
    origin_incarnation
    origin_sequence
    event_id
    event_kind
    payload_digest
    optional_hybrid_logical_time
    authority_epoch_or_zero
    authentication_tag
}
```

`(origin_node_id, origin_incarnation, origin_sequence)` is the causal dot and
deduplication identity. A process restart obtains a fresh random incarnation;
sequence numbers never continue ambiguously across incarnations. Hybrid logical
time is useful for traces and advisory recency, but is never a publication,
leadership, revocation, lease, or GC fence.

The authentication tag covers a domain-separated canonical fixed-binary
envelope and the payload digest. Protobuf is a transport encoding and is not
signed as raw serializer output. Events and inventory summaries are tenant
scoped and must not expose cross-tenant-stable content or prefix identifiers.

Initial event kinds should remain narrow:

- `ReplicaAdded` and `ReplicaRemoved`;
- `ManifestClosureVerified`;
- `NodeIncarnationObserved`;
- `AccessBucketPublished` and `CostObservationPublished`;
- `PublicationReceiptObserved`;
- `CacheWeaverEvidenceAdded`;
- authoritative, separately fenced configuration events where required.

Outbox retention is bounded by receiver acknowledgements and anti-entropy
checkpoints. A receiver that falls behind the retained event window repairs from
an inventory snapshot rather than requiring unbounded history.

## Replica directory CRDT

Replica discovery is the primary CRDT use case and belongs in `kvpack-cache`.
A location is an immutable advertisement with a unique causal dot:

```text
ReplicaAdded {
    tenant
    object_id
    node_id
    node_incarnation
    tier
    representation
    failure_domain
    publication_receipt
    dot
}
```

`ReplicaRemoved` removes only dots observed by the remover. Concurrent additions
therefore survive unrelated removals, while a removed physical incarnation
cannot be resurrected accidentally by message reordering. State-based snapshots
use a compact dotted version vector or per-origin high-water ranges; operation
replication uses the same dots. A global vector with one component per device is
not permitted because its metadata is unbounded at fleet scale.

Node restart creates a new incarnation. Advertisements from an older
incarnation become ineligible after the node authority accepts the new one.
Before that convergence, attempting an old advertisement is safe: the fetch
fails or the exact validator rejects it, and the request follows the existing
miss/recompute path.

A discovery row never authorizes restore, proves residency, or acts as a pin.
The selected source must still produce an owner-issued read/source lease and the
consumer must authenticate every object.

### Different meanings of removal

The control plane must not use one undifferentiated tombstone for all removal:

- `ReplicaEvicted` removes one physical location and permits a later re-add;
- `ArtifactSuperseded` advances a policy pointer without changing old bytes;
- `ArtifactRevoked` forbids future use for security, compliance, or invalid
  qualification;
- `EpochRetired` makes dependencies on an old cryptographic epoch ineligible;
- `PhysicallyReclaimed` records that one owner removed bytes after all safety
  barriers passed.

Only replica eviction is an ordinary observed-remove operation. Revocation,
epoch retirement, and production qualification are authoritative monotonic
epochs or signed pointers. A node unable to establish the required current epoch
must fail closed for that operation.

## Inventory summaries and anti-entropy

Prefix identity and durable manifests remain unchanged. Merkle trees, Bloom/XOR
filters, and invertible Bloom lookup tables are inventory tools above the object
format, not replacements for the exact input chain.

Each tenant and object-key shard periodically publishes a salted inventory
summary. Peers reconcile divergent ranges using:

- Merkle range summaries for large or unknown differences;
- an invertible Bloom lookup table when the expected difference is small;
- a bounded explicit list only after the summary identifies a small range.

The protocol applies bandwidth, CPU, battery, and privacy bounds before
reconciliation. A summary is only a candidate hint. It cannot reveal raw
prompts, tokens, responses, embeddings, unkeyed prefix IDs, or cross-tenant
residency.

For prefix lookup, tenant-scoped Bloom or XOR summaries may indicate which
regional shard or node is likely to have a cut. False positives cause a bounded
query followed by a miss; false negatives reduce hit rate but cannot weaken
correctness.

## Publication and replica transfer

Cross-node publication is an idempotent saga, not distributed two-phase commit:

1. Choose the destination by policy and failure domain.
2. Reserve destination quota/endurance under its authoritative owner.
3. Transfer missing chunks to private staging.
4. Authenticate, fsync, and publish each immutable chunk locally.
5. Transfer and authenticate the manifest.
6. Verify that the complete manifest dependency closure exists locally.
7. Commit the destination catalog transaction.
8. Emit `ReplicaAdded` only after that transaction commits.

A crash before step 7 leaves bounded orphan/private state for normal quarantine
or GC and never creates a searchable location. Repetition either verifies the
same immutable bytes or rejects a conflict. No distributed lock is needed for
the payload itself.

Durability classes define how many verified storage receipts are needed before
an artifact may be advertised as durable. Recomputable device caches may use one
local receipt and asynchronous replication. A durable gateway tier may require
receipts across configured failure domains. The manifest is advertised only
after all receipts required by its declared durability class exist.

Publication returns a causal session token:

```text
PublicationReceipt {
    tenant
    shard
    origin_node_id
    origin_incarnation
    committed_sequence
    manifest_id
    authority_epoch
}
```

A lookup carrying this receipt either waits for the requested sequence, queries
the origin, or deliberately returns a clean miss. This provides read-your-writes
for upload and promotion workflows without making every catalog lookup
linearizable.

## Placement and distributed singleflight

Use tenant-keyed weighted rendezvous hashing over eligible failure domains to
select a small deterministic home set for each immutable object. Dynamic load,
latency, proof, and remaining-work estimates choose among that set; they do not
change identity.

Demand fills use a short, fenced claim owned by the selected home shard. Waiters
watch the claim and may steal it after authoritative expiry. Exactly one network
fill is an optimization rather than a safety invariant: duplicate fills of an
immutable object are acceptable and converge during publication. A global
distributed mutex is therefore unnecessary.

Reads may be hedged among a bounded number of authenticated replicas when the
predicted latency benefit exceeds the extra network cost. Each losing attempt is
cancelled and releases its source lease independently.

## Membership and gossip

Membership observations are advisory. A SWIM-style failure detector can route
away from a suspected node and trigger replacement replication, but suspicion
must never:

- release an admission or engine-memory grant;
- terminate a source/read lease;
- authorize physical deletion;
- advance a key or qualification epoch;
- elect a gateway writer without the CP authority.

Full-fleet gossip and full-fleet vector clocks are forbidden. Large deployments
use a hierarchy:

```text
device-local agent
    strong local catalog, pins, grants, and engine ownership
        |
access/metro cohort
    AP replica advertisements and bounded membership gossip
        |
regional tenant shard
    placement, inventory anti-entropy, causal session service
        |
small CP authority
    gateway terms, keys, revocations, quota rights, qualification
```

Transient phones and workstations are soft replicas, never the sole durability
quorum for an artifact. Gossip cadence and replica eligibility account for
network class, battery, thermal state, data plan, and trust posture.

## Strong authority plane

The first production topology remains one active Mac/Iodyne gateway with an
exclusive instance lock. CRDT work must not delay that Qwen2.5 release.

Active/passive or multi-host gateway deployment additionally requires a small
Raft/etcd-equivalent authority that issues a monotonically increasing gateway
term. Every storage/catalog mutation carries the term, and the storage owner
rejects stale terms. Consensus contains only authority metadata; chunks,
manifests, ordinary lookups, telemetry, and popularity events do not pass through
the consensus log.

The strong authority owns:

- active gateway term and membership configuration;
- minimum readable and active key epochs;
- security/compliance revocations;
- hard quota-right allocation;
- representation/model qualification and production-policy pointer;
- any global guarantee concerning the last durable replica.

An authority outage leaves previously verified local data usable only where the
operation's epoch policy permits it. Operations requiring proof of the latest
revocation or key epoch fail closed. Ordinary cache discovery may still return a
miss and recompute.

## Quota and endurance rights

Do not merge independent byte counters with a PN-counter when a hard aggregate
limit matters; concurrent writers can each admit against the same remaining
capacity.

The first release should retain one authoritative reservation ledger. If future
gateways must admit durable writes while partitioned, use escrow/bounded-counter
rights:

```text
tenant durable budget = 100 TB
gateway A rights      =  20 TB
gateway B rights      =  20 TB
unallocated reserve   =  60 TB
```

A gateway spends only locally owned rights. Rights transfers are causally
tracked, idempotent, and cannot create new rights. The same mechanism can bound
physical write endurance in five-minute buckets. A partitioned writer eventually
stops admitting durable work when its local rights are exhausted; demand
restores and recomputation continue according to policy.

## Lease, time, and garbage-collection safety

Persisted wall time is useful for audit and restart recovery but is not a safe
exclusive fence. Runtime lease decisions use the grantor's monotonic/boot clock.
A durable lease records:

```text
lease_id
owner_id
owner_incarnation
authority_term
granted_wall_time
maximum_duration
object_id
state
```

On the same boot, the owner maps the duration to a monotonic deadline. After
restart or leadership change, unresolved leases from the previous incarnation
remain conservatively charged for a bounded recovery/grace interval. A forward
wall-clock jump may not cause early reclamation. Remote clients see an opaque
lease and term; their clocks do not decide expiry.

Physical GC is a protocol:

1. Commit a delete intent that prevents new leases under the old generation.
2. Drain owner-local pins, FDs, uploads, and active leases.
3. Wait any failover/grace barrier required by the lease term.
4. Rename to private trash and commit the local lifecycle transition.
5. Unlink, fsync the directory, and emit `PhysicallyReclaimed`.
6. Retain the revocation or epoch barrier for its required lifetime.

An open same-host FD continues to protect an unlinked file, but this local fact
does not replace source leases for remote streams.

## CacheWeaver and learned policy

CacheWeaver remains outside exact authorization. Its distributed state is best
represented as immutable events:

- exact-miss observations containing governed opaque identifiers;
- candidate and proposal identities;
- reconciler version and declared repair plan;
- bounded verification evidence;
- measured quality, cost, and failure results;
- provenance edges from an exact source to a derived artifact.

Concurrent evidence is retained rather than resolved with last-writer-wins. A
derived materialized view may rank candidates eventually consistently. Promotion
of a reconciler/model/policy into an approved production set, and revocation of
that approval, is a signed CP pointer. Approximation chaining and cross-tenant
candidate retrieval remain forbidden.

## Project ownership

| Project/component | Distributed-systems responsibility |
|---|---|
| `kvpack-core` and the local store | Exact immutable objects, authentication, local transactional publication, pins, and GC invariants |
| `kvpack-agent` | Strong local admission, peer ownership, shadow allocation, install/abort, and engine-free release |
| `kvpack-gateway` | Fenced storage writer, authenticated transfer, destination closure verification, and source leases |
| `kvpack-cache` | Replica CRDT, global discovery, placement, outbox/inbox transport, anti-entropy, causal receipts, and distributed-fill coordination |
| CacheWeaver | Append-only approximate evidence/provenance and rebuildable semantic/materialized indexes |
| KMS/configuration authority | Key/revocation epochs, gateway terms, quota rights, and production qualification |

No sister repository may create a second durable-byte validator or reinterpret a
failed exact validation as a usable candidate.

## Explicit anti-patterns

The following designs are rejected:

- one CRDT or one vector clock spanning the entire device fleet;
- last-writer-wins publication, deletion, or revocation based on wall time;
- OR-set admission grants, read leases, or engine-memory ownership;
- PN-counter enforcement of a hard quota;
- reclaiming resources or bytes because gossip suspects a process or node;
- merging SQLite databases or WALs as a replication protocol;
- putting chunk payloads or ordinary cache hits through Raft;
- distributed two-phase commit across payload replicas;
- a global lock for every content-addressed fill;
- cross-tenant deduplication, summaries, gossip, or residency disclosure;
- treating a location advertisement as restore authorization;
- advertising a manifest before its destination has verified complete closure.

## Verification and formal model

Before enabling multi-writer discovery or gateway failover, specify the following
state machines in TLA+/PlusCal or an equivalent executable formal model:

1. chunk-first replica publication and manifest advertisement;
2. gateway election, stale-term rejection, and failover;
3. lease acquisition, delete intent, drain, restart, and physical reclaim;
4. location add/remove, node reincarnation, and anti-entropy compaction;
5. revocation and key-epoch propagation under partition;
6. escrow-right spend, transfer, recovery, and exhaustion;
7. causal publication receipt followed by lookup on a lagging replica.

Safety properties include:

- no advertised destination lacks the manifest's verified dependency closure;
- no stale gateway term mutates authoritative storage;
- no live or uncertain protected object is physically reclaimed;
- no aggregate hard quota exceeds issued rights;
- no revoked or unqualified representation is installed;
- no approximate artifact is relabeled exact;
- partitions and duplicated/reordered events can reduce hit rate but cannot
  produce incorrect engine-visible state.

Implementation tests must inject network partitions, duplicate and reordered
events, lost acknowledgements, clock jumps, node reincarnation, leader
failover, catalog corruption, disk full, and delayed anti-entropy. A
Jepsen-style history checker should validate the CP operations, while property
tests validate CRDT convergence under arbitrary delivery order.

## Delivery order

### Before the single-gateway Qwen2.5 release

1. Freeze this consistency matrix and name the linearization point for every
   publication, grant, lease, install, revocation, and GC operation.
2. Split eviction, supersession, revocation, epoch retirement, and physical
   reclamation in catalog types and schemas.
3. Define the canonical event envelope, transactional outbox/inbox, origin
   incarnation, and causal publication receipt.
4. Replace wall-clock-only lease/GC decisions with owner-monotonic deadlines,
   restart uncertainty, and explicit fencing terms.
5. Scope every scalar: key epoch, catalog/configuration epoch, publication
   generation, gateway term, node incarnation, origin sequence, and observation
   time are distinct types.
6. Add the formal models and local fault-injection cases. Distributed features
   may remain disabled.

### After the single-gateway baseline

1. Implement the `kvpack-cache` location observed-remove set.
2. Add tenant-sharded anti-entropy and causal session behavior.
3. Add weighted rendezvous placement and best-effort distributed singleflight.
4. Add bounded regional membership gossip and advisory load observations.
5. Add active/passive gateway failover with storage-enforced fencing.

### Only when disconnected durable writers are required

1. Introduce escrow quota/endurance rights.
2. Qualify rights recovery and transfer under partition and node loss.
3. Add cold-tier erasure coding only where its repair traffic and latency beat
   ordinary replication; device caches remain replicated/recomputable soft state.

## Research basis

- Hellerstein and Alvaro, *Keeping CALM: When Distributed Consistency is Easy*:
  <https://arxiv.org/abs/1901.01930>
- Bailis et al., *Coordination Avoidance in Database Systems*:
  <https://www.vldb.org/pvldb/vol8/p185-bailis.pdf>
- Shapiro et al., *Conflict-Free Replicated Data Types*:
  <https://people.eecs.berkeley.edu/~kubitron/courses/cs262a-F19/handouts/papers/Shapiro-CRDT.pdf>
- Balegas et al., *Extending Eventually Consistent Cloud Databases for Enforcing
  Numeric Invariants*:
  <https://arxiv.org/abs/1503.09052>
- Das, Gupta, and Motivala, *SWIM: Scalable Weakly-consistent
  Infection-style Process Group Membership Protocol*:
  <https://www.cs.cornell.edu/projects/quicksilver/public_pdfs/SWIM.pdf>
- Kulkarni, Demirbas, Madeppa, Avva, and Leone, *Logical Physical Clocks and
  Consistent Snapshots in Globally Distributed Databases*:
  <https://cse.buffalo.edu/~demirbas/publications/hlc.pdf>
- Gray and Cheriton, *Leases: An Efficient Fault-Tolerant Mechanism for
  Distributed File Cache Consistency*:
  <https://courses.cs.duke.edu/compsci510/spring15/readings/leases-sosp89.pdf>
- Ongaro and Ousterhout, *In Search of an Understandable Consensus Algorithm*:
  <https://raft.github.io/raft.pdf>
- Goodrich and Mitzenmacher, *Invertible Bloom Lookup Tables*:
  <https://arxiv.org/abs/1101.2245>
