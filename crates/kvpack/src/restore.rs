use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use kvpack_core::{
    decode_authenticated_pack, decode_chunk, decode_chunk_with_stats, inspect_pack_header,
    representation_family_id, semantic_model_id, validate_family, ChunkRef, CutManifest, Id32,
    InputCutId, ManifestKind, RealizedCutSchemaId, RepresentationFamilyId, SemanticModelId,
    StateDeclaration, StateKey, StatsSidecar, ValidationContext, MAX_CHUNKS_PER_STATE,
    MAX_DELTA_DEPTH,
};
use sha2::{Digest, Sha256};

use crate::store::{family_digest, semantic_digest, RetainedPin};
use crate::telemetry::{
    ByteCounter, OpaqueSpanId, ServiceComponent, SpanOutcome, TraceContext, TracePhase,
};
use crate::{LocalStore, RestoreStatePlan, StoreError, VerifiedRestoreSink};

type StoredChunkAvailabilityRowTuple = (Vec<u8>, Vec<u8>, u64, u32, u32, u64, String);

/// One catalog chunk row matched against an authenticated chunk reference.
#[derive(Debug, Clone)]
pub(super) struct StoredChunkAvailabilityRow {
    pub(super) chunk_id: Vec<u8>,
    pub(super) object_digest: Vec<u8>,
    pub(super) key_epoch: u64,
    pub(super) plaintext_bytes: u32,
    pub(super) object_bytes: u32,
    pub(super) fidelity_rung: u64,
    pub(super) location_state: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RestoreTier {
    Local,
    Gateway,
    Resident,
    Recompute,
}

#[derive(Debug, Clone)]
pub struct RestoreRequest {
    pub semantic_model: SemanticModelId,
    pub family: RepresentationFamilyId,
    pub input_tokens: Vec<u32>,
    pub auxiliary_inputs: Vec<kvpack_core::AuxiliaryInputId>,
    pub minimum_key_epoch: u64,
    /// Total returned candidates including the explicit recomputation plan.
    pub maximum_candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RestoreResourceRequirements {
    pub shadow_bytes: u64,
    pub pinned_source_bytes: u64,
    pub scratch_bytes_per_task: u64,
    pub staging_bytes: u64,
    pub receive_window_bytes: u64,
    pub safety_margin_bytes: u64,
    pub source_pins: u64,
    pub source_fds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeldRestoreResources {
    pub active_restores: u64,
    pub shadow_bytes: u64,
    pub pinned_source_bytes: u64,
    pub scratch_bytes: u64,
    pub staging_bytes: u64,
    pub receive_window_bytes: u64,
    pub safety_margin_bytes: u64,
    pub source_pins: u64,
    pub source_fds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RestoreResourceCharge {
    pub(crate) shadow_bytes: u64,
    pub(crate) pinned_source_bytes: u64,
    pub(crate) scratch_bytes: u64,
    pub(crate) staging_bytes: u64,
    pub(crate) receive_window_bytes: u64,
    pub(crate) safety_margin_bytes: u64,
    pub(crate) source_pins: u64,
    pub(crate) source_fds: u64,
}

impl RestoreResourceCharge {
    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            shadow_bytes: self.shadow_bytes.checked_add(other.shadow_bytes)?,
            pinned_source_bytes: self
                .pinned_source_bytes
                .checked_add(other.pinned_source_bytes)?,
            scratch_bytes: self.scratch_bytes.checked_add(other.scratch_bytes)?,
            staging_bytes: self.staging_bytes.checked_add(other.staging_bytes)?,
            receive_window_bytes: self
                .receive_window_bytes
                .checked_add(other.receive_window_bytes)?,
            safety_margin_bytes: self
                .safety_margin_bytes
                .checked_add(other.safety_margin_bytes)?,
            source_pins: self.source_pins.checked_add(other.source_pins)?,
            source_fds: self.source_fds.checked_add(other.source_fds)?,
        })
    }
}

/// An exact, path-free availability hint from an engine-resident cache or a
/// gateway discovery result. Hints only generate plans; they never authorize
/// installation, and the selected source must still verify its bytes/handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreAvailableSource {
    Resident {
        matched_cut: InputCutId,
        resident_identity: Id32,
        restored_bytes: u64,
        resources: RestoreResourceRequirements,
    },
    Gateway {
        matched_cut: InputCutId,
        manifest_id: Id32,
        chain_identity: Id32,
        key_epoch: u64,
        restored_bytes: u64,
        resources: RestoreResourceRequirements,
    },
}

#[derive(Debug, Clone)]
struct AuthenticatedManifest {
    manifest_id: Id32,
    manifest: Arc<CutManifest>,
}

#[derive(Debug, Clone)]
pub struct RestoreCandidate {
    tenant_namespace: Id32,
    matched_cut: InputCutId,
    requested_cut: InputCutId,
    suffix_tokens: u64,
    tier: RestoreTier,
    manifest_id: Option<Id32>,
    chain_identity: Option<Id32>,
    source_identity: Option<Id32>,
    source_key_epoch: Option<u64>,
    restored_bytes: u64,
    resources: RestoreResourceRequirements,
    semantic_id: Id32,
    family_id: Id32,
    minimum_key_epoch: u64,
    // Chunks on the tombstone fidelity rung: planned as guided-recompute
    // candidates, never served as bytes.
    tombstoned_chunks: u64,
}

impl RestoreCandidate {
    pub fn matched_cut(&self) -> InputCutId {
        self.matched_cut
    }

    pub fn requested_cut(&self) -> InputCutId {
        self.requested_cut
    }

    pub fn suffix_tokens(&self) -> u64 {
        self.suffix_tokens
    }

    pub fn tier(&self) -> RestoreTier {
        self.tier
    }

    pub fn manifest_id(&self) -> Option<Id32> {
        self.manifest_id
    }

    pub fn chain_identity(&self) -> Option<Id32> {
        self.chain_identity
    }

    pub fn source_identity(&self) -> Option<Id32> {
        self.source_identity
    }

    pub fn source_key_epoch(&self) -> Option<u64> {
        self.source_key_epoch
    }

    pub fn restored_bytes(&self) -> u64 {
        self.restored_bytes
    }

    pub fn resources(&self) -> RestoreResourceRequirements {
        self.resources
    }

    /// Number of referenced chunks on the tombstone fidelity rung.  Their
    /// local bytes were dropped by demotion; the restore plan marks them as
    /// guided-recompute candidates and never serves bytes for them.
    pub fn tombstoned_chunks(&self) -> u64 {
        self.tombstoned_chunks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreLimits {
    pub maximum_shadow_bytes: u64,
    pub maximum_pinned_source_bytes: u64,
    pub maximum_scratch_bytes: u64,
    pub maximum_staging_bytes: u64,
    pub maximum_receive_window_bytes: u64,
    pub maximum_safety_margin_bytes: u64,
    pub maximum_source_pins: u64,
    pub maximum_source_fds: u64,
    pub maximum_parallelism: usize,
}

impl Default for RestoreLimits {
    fn default() -> Self {
        Self {
            maximum_shadow_bytes: 4 * 1024 * 1024 * 1024 * 1024,
            maximum_pinned_source_bytes: 4 * 1024 * 1024 * 1024 * 1024,
            maximum_scratch_bytes: 256 * 1024 * 1024,
            maximum_staging_bytes: 4 * 1024 * 1024 * 1024 * 1024,
            maximum_receive_window_bytes: 256 * 1024 * 1024,
            maximum_safety_margin_bytes: 256 * 1024 * 1024,
            maximum_source_pins: 1_000_000,
            maximum_source_fds: 1_000_000,
            maximum_parallelism: 4,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RestoreCancellation {
    cancelled: Arc<AtomicBool>,
}

impl RestoreCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone)]
struct RestoreChunkOperation {
    state_key: StateKey,
    span: kvpack_core::ChunkSpan,
    /// Byte offset in the compact destination plane. This normally equals
    /// the authenticated span offset; TailWindow snapshots subtract their
    /// authenticated absolute logical start because only the tail is stored.
    target_offset: u64,
    reference: kvpack_core::ChunkRef,
    tenant_namespace: Id32,
    family: RepresentationFamilyId,
    /// Tombstone-rung chunk: a guided-recompute candidate.  The plan carries
    /// the marker (chained key + token cut span); bytes are never served.
    recompute: bool,
}

pub struct AuthenticatedRestorePlan {
    store: Arc<LocalStore>,
    manifest_id: Id32,
    semantic_model: SemanticModelId,
    family: RepresentationFamilyId,
    realized_schema: RealizedCutSchemaId,
    key_epoch: u64,
    matched_cut: InputCutId,
    requested_cut: InputCutId,
    chain_identity: Id32,
    states: Vec<RestoreStatePlan>,
    operations: Vec<RestoreChunkOperation>,
    resources: RestoreResourceRequirements,
    limits: RestoreLimits,
}

pub const MAX_SCATTER_FDS_PER_BATCH: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedScatterDescriptor {
    pub manifest_id: Id32,
    pub state_key: StateKey,
    pub chunk_ordinal: u64,
    pub batch_number: u32,
    pub fd_index: u32,
    pub fd_offset: u64,
    pub fd_bytes: u64,
    pub object_key: Id32,
    pub object_digest: Id32,
    pub object_bytes: u64,
    pub plaintext_bytes: u64,
    pub key_epoch: u64,
    pub target_offset: u64,
    pub target_bytes: u64,
    pub atomic_group: u32,
    pub attempt: Id32,
    pub descriptor_digest: Id32,
}

impl AuthenticatedScatterDescriptor {
    pub fn verify_digest(&self) -> bool {
        self.descriptor_digest == scatter_descriptor_digest(self)
    }
}

pub struct PinnedScatterBatch {
    batch_number: u32,
    total_batches: u32,
    descriptors: Vec<AuthenticatedScatterDescriptor>,
    files: Vec<File>,
    pins: Vec<RetainedPin>,
}

impl PinnedScatterBatch {
    pub fn batch_number(&self) -> u32 {
        self.batch_number
    }

    pub fn total_batches(&self) -> u32 {
        self.total_batches
    }

    pub fn descriptors(&self) -> &[AuthenticatedScatterDescriptor] {
        &self.descriptors
    }

    pub fn files(&self) -> &[File] {
        &self.files
    }

    pub fn pin_ids(&self) -> Result<Vec<Id32>, StoreError> {
        self.pins.iter().map(RetainedPin::id).collect()
    }
}

pub struct PreparedScatterTransfer {
    manifest_id: Id32,
    attempt: Id32,
    resources: RestoreResourceRequirements,
    batches: Vec<PinnedScatterBatch>,
}

impl PreparedScatterTransfer {
    pub fn manifest_id(&self) -> Id32 {
        self.manifest_id
    }

    pub fn attempt(&self) -> Id32 {
        self.attempt
    }

    pub fn resources(&self) -> RestoreResourceRequirements {
        self.resources
    }

    pub fn batches(&self) -> &[PinnedScatterBatch] {
        &self.batches
    }

    pub fn batch(&self, number: u32) -> Option<&PinnedScatterBatch> {
        self.batches.get(number as usize)
    }

    pub fn pin_ids(&self) -> Result<Vec<Id32>, StoreError> {
        self.batches
            .iter()
            .flat_map(|batch| &batch.pins)
            .map(RetainedPin::id)
            .collect()
    }
}

mod helpers;
mod plan;
mod shadow;

pub use shadow::ShadowRestoreHandle;

use helpers::*;
pub(crate) use helpers::{authenticated_chain_for_compaction, charge_within_limits};
struct RestoreReservation<'a> {
    store: &'a LocalStore,
    restore_id: Id32,
    retained: bool,
}

impl<'a> RestoreReservation<'a> {
    fn new(store: &'a LocalStore, restore_id: Id32) -> Self {
        Self {
            store,
            restore_id,
            retained: false,
        }
    }

    fn retain_until_engine_free(&mut self) {
        self.retained = true;
    }
}

impl Drop for RestoreReservation<'_> {
    fn drop(&mut self) {
        if !self.retained {
            let _ = self.store.acknowledge_engine_free(&self.restore_id);
        }
    }
}

#[must_use = "restore pins remain held until engine_free or explicit store acknowledgement"]
#[derive(Debug)]
pub struct InstalledRestore {
    store: Arc<LocalStore>,
    restore_id: Id32,
    manifest_id: Id32,
    trace_context: Option<TraceContext>,
    release_parent_span: Option<OpaqueSpanId>,
    engine_freed: bool,
}

impl InstalledRestore {
    pub fn restore_id(&self) -> Id32 {
        self.restore_id
    }

    pub fn manifest_id(&self) -> Id32 {
        self.manifest_id
    }

    pub fn engine_free(mut self) -> Result<(), StoreError> {
        let started = Instant::now();
        let started_unix_ns = restore_now_ns();
        let released = self.store.acknowledge_engine_free(&self.restore_id);
        let outcome = match &released {
            Ok(true) => SpanOutcome::Ok,
            Ok(false) => SpanOutcome::Rejected,
            Err(StoreError::Integrity(_) | StoreError::Authentication(_)) => {
                SpanOutcome::IntegrityError
            }
            Err(StoreError::Cancelled) => SpanOutcome::Cancelled,
            Err(_) => SpanOutcome::Unavailable,
        };
        let _ = self
            .store
            .telemetry
            .observe_latency(TracePhase::Release, started.elapsed());
        if let Some(context) = self.trace_context.as_ref() {
            let _ = self.store.telemetry.record_span(
                context,
                self.release_parent_span,
                TracePhase::Release,
                outcome,
                started_unix_ns,
                restore_now_ns().max(started_unix_ns),
            );
        }
        if !released? {
            return Err(StoreError::State(
                "restore resources were already released or are unknown",
            ));
        }
        self.engine_freed = true;
        Ok(())
    }
}

fn restore_now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

impl Drop for InstalledRestore {
    fn drop(&mut self) {
        // Uncertainty retains pins. `acknowledge_engine_free` is deliberately
        // explicit and can be called with `restore_id` after handle loss.
        let _ = self.engine_freed;
    }
}

impl LocalStore {
    pub fn restore_candidates(
        self: &Arc<Self>,
        request: RestoreRequest,
    ) -> Result<Vec<RestoreCandidate>, StoreError> {
        self.restore_candidates_with_sources(request, &[])
    }

    pub fn restore_candidates_with_sources(
        self: &Arc<Self>,
        request: RestoreRequest,
        available_sources: &[RestoreAvailableSource],
    ) -> Result<Vec<RestoreCandidate>, StoreError> {
        if request.minimum_key_epoch == 0 || !(2..=65).contains(&request.maximum_candidates) {
            return Err(StoreError::Expectation(
                "restore request epoch or candidate bound is invalid",
            ));
        }
        if available_sources.len() > 256 {
            return Err(StoreError::Expectation(
                "restore availability hint count exceeds bound",
            ));
        }
        let minimum_key_epoch = request
            .minimum_key_epoch
            .max(self.minimum_readable_key_epoch());
        validate_family(&request.family)?;
        let keys = self.schedule(self.key_epoch())?;
        let (requested_cut, nodes) = kvpack_core::derive_input_cut(
            keys.prefix_key(),
            &self.tenant_namespace(),
            &request.semantic_model,
            &request.family,
            &request.input_tokens,
            &request.auxiliary_inputs,
        )?;
        let (zero_cut, _) = kvpack_core::derive_input_cut(
            keys.prefix_key(),
            &self.tenant_namespace(),
            &request.semantic_model,
            &request.family,
            &[],
            &request.auxiliary_inputs,
        )?;
        let semantic_id = semantic_digest(&request.semantic_model);
        let family_id = family_digest(&request.family)?;
        let expected_cuts: BTreeMap<_, _> = nodes
            .iter()
            .map(|node| {
                (
                    node.token_count,
                    InputCutId {
                        token_root: node.id,
                        auxiliary_input_root: requested_cut.auxiliary_input_root,
                        token_count: node.token_count,
                    },
                )
            })
            .collect();
        let hits = self.resolve_prefix_candidates_for_cut(
            &nodes,
            &semantic_id,
            &family_id,
            request.maximum_candidates - 1,
            requested_cut.token_count,
            minimum_key_epoch,
        )?;
        let validation_context = ValidationContext::default();
        let mut candidates = Vec::with_capacity(
            request
                .maximum_candidates
                .saturating_add(available_sources.len()),
        );
        // Immutable parents and chunk locations overlap heavily across adjacent
        // prefix candidates. Share validation only within this lookup so every
        // later request still observes missing or externally changed files.
        let mut authenticated_manifest_cache = BTreeMap::new();
        let mut local_availability_cache = BTreeMap::new();
        for hit in hits {
            let Some(matched_cut) = expected_cuts.get(&hit.token_count).copied() else {
                return Err(StoreError::Authentication(
                    "catalog candidate token count is not a requested prefix node",
                ));
            };
            let chain = match authenticate_chain_cached(
                self,
                hit.manifest_id,
                minimum_key_epoch,
                Some((semantic_id, family_id, matched_cut)),
                &validation_context,
                &mut authenticated_manifest_cache,
            ) {
                Ok(chain) => chain,
                Err(StoreError::NotFound) => continue,
                Err(StoreError::Io { source, .. })
                    if source.kind() == std::io::ErrorKind::NotFound =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            };
            let disposition =
                chain_chunk_disposition_cached(self, &chain, &mut local_availability_cache)?;
            if !disposition.complete {
                continue;
            }
            let resources = chain_resources(&chain)?;
            let source_key_epoch = chain.last().unwrap().manifest.key_epoch;
            candidates.push(RestoreCandidate {
                tenant_namespace: self.tenant_namespace(),
                matched_cut,
                requested_cut,
                suffix_tokens: requested_cut
                    .token_count
                    .saturating_sub(matched_cut.token_count),
                tier: RestoreTier::Local,
                manifest_id: Some(hit.manifest_id),
                chain_identity: Some(chain_identity(&chain)),
                source_identity: Some(hit.manifest_id),
                source_key_epoch: Some(source_key_epoch),
                restored_bytes: chain
                    .last()
                    .unwrap()
                    .manifest
                    .realized_schema
                    .complete_restored_bytes,
                resources,
                semantic_id,
                family_id,
                minimum_key_epoch,
                tombstoned_chunks: disposition.tombstoned.len() as u64,
            });
        }
        for source in available_sources {
            let (
                tier,
                matched_cut,
                manifest_id,
                chain_identity,
                source_identity,
                source_key_epoch,
                restored_bytes,
                resources,
            ) = match source {
                RestoreAvailableSource::Resident {
                    matched_cut,
                    resident_identity,
                    restored_bytes,
                    resources,
                } => (
                    RestoreTier::Resident,
                    *matched_cut,
                    None,
                    Some(*resident_identity),
                    Some(*resident_identity),
                    None,
                    *restored_bytes,
                    *resources,
                ),
                RestoreAvailableSource::Gateway {
                    matched_cut,
                    manifest_id,
                    chain_identity,
                    key_epoch,
                    restored_bytes,
                    resources,
                } => {
                    if *key_epoch < minimum_key_epoch || *key_epoch > self.key_epoch() {
                        continue;
                    }
                    (
                        RestoreTier::Gateway,
                        *matched_cut,
                        Some(*manifest_id),
                        Some(*chain_identity),
                        Some(*manifest_id),
                        Some(*key_epoch),
                        *restored_bytes,
                        *resources,
                    )
                }
            };
            if expected_cuts.get(&matched_cut.token_count) != Some(&matched_cut)
                || matched_cut.token_count == 0
                || restored_bytes == 0
                || source_identity == Some([0; 32])
                || chain_identity == Some([0; 32])
                || resource_charge(resources, 1).is_err()
                || (tier == RestoreTier::Gateway && resources.shadow_bytes != restored_bytes)
            {
                continue;
            }
            candidates.push(RestoreCandidate {
                tenant_namespace: self.tenant_namespace(),
                matched_cut,
                requested_cut,
                suffix_tokens: requested_cut
                    .token_count
                    .saturating_sub(matched_cut.token_count),
                tier,
                manifest_id,
                chain_identity,
                source_identity,
                source_key_epoch,
                restored_bytes,
                resources,
                semantic_id,
                family_id,
                minimum_key_epoch,
                tombstoned_chunks: 0,
            });
        }
        candidates.sort_by(|left, right| {
            right
                .matched_cut
                .token_count
                .cmp(&left.matched_cut.token_count)
                .then_with(|| tier_order(left.tier).cmp(&tier_order(right.tier)))
                .then_with(|| left.source_identity.cmp(&right.source_identity))
        });
        candidates.dedup_by(|left, right| {
            left.tier == right.tier
                && left.matched_cut == right.matched_cut
                && left.source_identity == right.source_identity
        });
        candidates.truncate(request.maximum_candidates - 1);
        candidates.push(RestoreCandidate {
            tenant_namespace: self.tenant_namespace(),
            matched_cut: zero_cut,
            requested_cut,
            suffix_tokens: requested_cut.token_count,
            tier: RestoreTier::Recompute,
            manifest_id: None,
            chain_identity: None,
            source_identity: None,
            source_key_epoch: None,
            restored_bytes: 0,
            resources: RestoreResourceRequirements::default(),
            semantic_id,
            family_id,
            minimum_key_epoch,
            tombstoned_chunks: 0,
        });
        Ok(candidates)
    }
}
