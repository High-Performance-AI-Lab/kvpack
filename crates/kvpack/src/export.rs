use std::io::Read;
use std::sync::Arc;

use kvpack_core::{
    chunk_id, encode_authenticated_pack, encode_chunk_with_content_id, representation_family_id,
    semantic_model_id, validate_family, AuxiliaryInputId, ChunkEncoding, ChunkObject, ChunkRef,
    ChunkSpan, CutManifest, EncodedPack, FamilyState, Id32, InputCutId, KeySchedule, Layout,
    ManifestKind, PrefixNode, RealizedCutSchemaId, RepresentationFamilyId, SemanticModelId, Shape,
    StateKey, MAX_CHUNKS_PER_STATE, MAX_CHUNK_PLAINTEXT, MAX_DELTA_DEPTH, PREFIX_BLOCK_TOKENS,
};

use crate::intent::IntentHasher;
use crate::store::{PendingManifest, UploadReservation, CHUNK_PUT_BATCH_BYTES};
use crate::{ByteCounter, LocalStore, PublishedArtifact, StoreError, UploadState, WritePolicy};

mod manifests;
mod plan;
mod provisional;
mod session;
mod writer;

use plan::{
    estimate_export_reservation, export_intent_digest, family_bytes_per_token, next_chunk_bytes,
    physical_footprint, shape_for_cut, strides_for_cut, validate_export_declaration,
};
pub use provisional::{
    ProvisionalExportDeclaration, ProvisionalExportReceipt, ProvisionalExportSeal,
    ProvisionalExportSession, ProvisionalIoIntervalV1, ProvisionalStateReceipt,
};
pub use session::ExportSession;
use session::{published_cut, ExportStagedChunk, StoredState, StoredStateChunk};
pub use writer::ExportStateWriter;

/// The immutable production-v1 cut policy. Requests cannot widen it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportCutPolicy {
    checkpoint_tokens: u64,
}

impl ExportCutPolicy {
    pub const fn production_v1() -> Self {
        Self {
            checkpoint_tokens: PREFIX_BLOCK_TOKENS as u64,
        }
    }

    pub const fn checkpoint_tokens(self) -> u64 {
        self.checkpoint_tokens
    }
}

impl Default for ExportCutPolicy {
    fn default() -> Self {
        Self::production_v1()
    }
}

/// Engine-owned metadata for one complete logical state stream. Object IDs,
/// roots, realized schemas, parent depths, and paths are deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportStateDeclaration {
    pub key: StateKey,
    pub strides: Vec<u64>,
    pub atomic_group: u32,
}

/// Complete export metadata declared before any state source is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportDeclaration {
    pub semantic_model: SemanticModelId,
    pub input_tokens: Vec<u32>,
    pub auxiliary_inputs: Vec<AuxiliaryInputId>,
    pub family: RepresentationFamilyId,
    pub states: Vec<ExportStateDeclaration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportStateBounds {
    pub token_count: u64,
    pub bytes_per_token: u64,
    pub plaintext_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedCut {
    pub manifest_id: Id32,
    pub input_cut: InputCutId,
    pub realized_schema: RealizedCutSchemaId,
    pub restored_bytes: u64,
    pub publication_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedCutSet {
    pub checkpoints: Vec<PublishedCut>,
    pub exact_final: PublishedCut,
}
