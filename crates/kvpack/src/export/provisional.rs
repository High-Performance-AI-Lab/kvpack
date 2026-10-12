use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, Instant};

use kvpack_core::{
    chunk_id, encode_authenticated_pack, encode_chunk_with_content_id, manifest_id,
    representation_family_id, semantic_model_id, AuxiliaryInputId, ChunkEncoding, ChunkObject,
    ChunkRef, ChunkSpan, EncodedPack, Id32, KeySchedule, ManifestKind, RepresentationFamilyId,
    SemanticModelId, StateKey, MAX_CHUNKS_PER_STATE, MAX_CHUNK_PLAINTEXT, MAX_DELTA_DEPTH,
};
use sha2::{Digest, Sha256};

use super::manifests::{manifest_for_cut_parts, ManifestParts};
use super::{
    estimate_export_reservation, family_bytes_per_token, next_chunk_bytes, published_cut,
    validate_export_declaration, ExportCutPolicy, ExportDeclaration, ExportStateBounds,
    ExportStateDeclaration, PublishedCutSet, StoredState, StoredStateChunk,
};
use crate::intent::IntentHasher;
use crate::store::{
    PendingManifest, ProvisionalPromotedChunk, ProvisionalProvenance, ProvisionalStageMode,
    ProvisionalStagedChunk, UploadReservation, CHUNK_PUT_BATCH_BYTES,
};
use crate::{portable_prefill_token_ids_sha256, LocalStore, StoreError, UploadState, WritePolicy};

mod helpers;
mod seal;
mod session;
mod stage;

use helpers::{
    duration_ns, elapsed_ns, ensure_source_ended, provisional_intent_digest,
    provisional_seal_digest, read_exact_object, read_exact_state,
};
use session::ProvisionalChunk;
pub use session::ProvisionalExportSession;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionalExportDeclaration {
    pub semantic_model: SemanticModelId,
    pub cached_token_count: u32,
    pub sealed_prompt_token_ids_sha256: Id32,
    pub source_declaration_digest: Id32,
    pub auxiliary_inputs: Vec<AuxiliaryInputId>,
    pub family: RepresentationFamilyId,
    pub states: Vec<ExportStateDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionalExportSeal {
    pub prompt_token_ids: Vec<u32>,
    pub artifact_digest: Id32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvisionalIoIntervalV1 {
    pub start_ns: u64,
    pub end_ns: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionalStateReceipt {
    pub state: StateKey,
    pub duration_ns: u64,
    pub plaintext_bytes: u64,
    pub chunk_count: u64,
    pub staged_bytes: u64,
    pub deduplicated_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProvisionalExportReceipt {
    pub begin_duration_ns: u64,
    pub encryption_duration_ns: u64,
    pub staging_duration_ns: u64,
    pub promotion_duration_ns: u64,
    pub publication_duration_ns: u64,
    pub total_duration_ns: u64,
    pub staged_bytes: u64,
    pub deduplicated_bytes: u64,
    pub promoted_bytes: u64,
    pub staged_chunk_count: u64,
    pub deduplicated_chunk_count: u64,
    pub promoted_chunk_count: u64,
    pub chunk_count: u64,
    /// The prompt token immediately past the durable cut (the +1 decode
    /// boundary token). Also persisted on the upload row for the decode side.
    pub boundary_token_id: u32,
    /// Provenance captured at begin and hashed into the seal digest.
    pub provenance: ProvisionalProvenance,
    pub write_intervals: Vec<ProvisionalIoIntervalV1>,
    pub encryption_intervals: Vec<ProvisionalIoIntervalV1>,
    pub published: PublishedCutSet,
}
