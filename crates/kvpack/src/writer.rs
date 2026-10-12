use std::collections::BTreeMap;
use std::sync::Arc;

use kvpack_core::{
    chunk_id, decode_authenticated_pack, derive_input_cut, encode_authenticated_pack,
    encode_chunk_with_content_id, inspect_pack_header, representation_family_id, semantic_model_id,
    validate_family, AtomicGroup, ChunkEncoding, ChunkObject, ChunkRef, ChunkSpan, CutManifest,
    Id32, InputCutId, KeySchedule, ManifestDeclaration, ManifestKind, PrefixNode,
    RealizedCutSchemaId, RealizedStateSchema, StateDeclaration, StateKey, StateManifest,
    StaticDimension, TokenAxisRule, ValidationContext, MAX_CHUNKS_PER_STATE, MAX_CHUNK_PLAINTEXT,
    MAX_DELTA_DEPTH,
};

use crate::intent::IntentHasher;
use crate::restore::authenticated_chain_for_compaction;
use crate::store::{UploadReservation, CHUNK_PUT_BATCH_BYTES};
use crate::{LocalStore, StoreError, UploadState};

#[derive(Debug, Clone)]
pub struct WritePolicy {
    pub(crate) idempotency_key: Id32,
    pub(crate) encrypt_chunks: bool,
    pub(crate) encrypt_manifest: bool,
    pub(crate) qualified_semantic_models: Vec<kvpack_core::SemanticModelId>,
    pub(crate) qualified_representation_families: Vec<Id32>,
    pub(crate) maximum_restored_bytes: u64,
    pub(crate) publication_generation: u64,
    pub(crate) retention: crate::RetentionInputs,
    /// M7 opt-in: derive an authenticated statistics sidecar from every
    /// written fp16 state plane, retaining this many top sink scores.
    pub(crate) stats_sidecar_sink_count: Option<u8>,
}

impl WritePolicy {
    pub fn exact_qualified(
        idempotency_key: Id32,
        semantic: kvpack_core::SemanticModelId,
        family: &kvpack_core::RepresentationFamilyId,
    ) -> Result<Self, StoreError> {
        Ok(Self {
            idempotency_key,
            encrypt_chunks: true,
            encrypt_manifest: true,
            qualified_semantic_models: vec![semantic],
            qualified_representation_families: vec![representation_family_id(family)?],
            maximum_restored_bytes: 4 * 1024 * 1024 * 1024 * 1024,
            publication_generation: 1,
            retention: crate::RetentionInputs::default(),
            stats_sidecar_sink_count: None,
        })
    }

    pub fn with_publication_generation(
        mut self,
        publication_generation: u64,
    ) -> Result<Self, StoreError> {
        if publication_generation == 0 || publication_generation > i64::MAX as u64 {
            return Err(StoreError::State("invalid publication generation"));
        }
        self.publication_generation = publication_generation;
        Ok(self)
    }

    /// Select the closed v1 plaintext or encrypted representation. This does
    /// not alter semantic/family qualification or any durable identity.
    pub fn with_encryption(mut self, enabled: bool) -> Self {
        self.encrypt_chunks = enabled;
        self.encrypt_manifest = enabled;
        self
    }

    /// Narrow the default restored-byte ceiling for one export. Requests may
    /// not raise the immutable production ceiling.
    pub fn with_maximum_restored_bytes(
        mut self,
        maximum_restored_bytes: u64,
    ) -> Result<Self, StoreError> {
        const PRODUCTION_MAXIMUM: u64 = 4 * 1024 * 1024 * 1024 * 1024;
        if maximum_restored_bytes == 0 || maximum_restored_bytes > PRODUCTION_MAXIMUM {
            return Err(StoreError::Expectation(
                "writer restored-byte bound cannot widen production policy",
            ));
        }
        self.maximum_restored_bytes = maximum_restored_bytes;
        Ok(self)
    }

    pub fn with_retention(mut self, retention: crate::RetentionInputs) -> Result<Self, StoreError> {
        self.retention = retention.validate()?;
        Ok(self)
    }

    /// Opt into the M7 statistics sidecar: every fp16 state plane written
    /// through this policy gets an authenticated per-chunk sidecar derived
    /// from the state bytes (per-channel K min/max, per-token key L2 norms,
    /// top-`sink_count` sink scores).  Chunks deduplicated against an
    /// already-durable object keep the existing object's form.  Non-fp16
    /// states and shapes beyond the sidecar bounds fail the write closed.
    pub fn with_stats_sidecars(mut self, sink_count: u8) -> Result<Self, StoreError> {
        if sink_count == 0 || sink_count as usize > kvpack_core::MAX_SINK_SCORES {
            return Err(StoreError::Expectation(
                "stats sidecar sink count is outside bounded limits",
            ));
        }
        self.stats_sidecar_sink_count = Some(sink_count);
        Ok(self)
    }

    pub(crate) fn validation_context(&self) -> ValidationContext {
        let mut context = ValidationContext::default();
        context.bounds.max_restored_bytes = self.maximum_restored_bytes;
        context
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishedArtifact {
    pub manifest_id: Id32,
    pub tenant_namespace: Id32,
    pub restored_bytes: u64,
    pub publication_generation: u64,
}

pub struct ArtifactWriter {
    store: Arc<LocalStore>,
    declaration: ManifestDeclaration,
    input_cut: InputCutId,
    exact_prefix_node: PrefixNode,
    policy: WritePolicy,
    keys: KeySchedule,
    completed: Vec<StateManifest>,
    next_state: usize,
    poisoned: bool,
    committed: bool,
    replayed: Option<PublishedArtifact>,
    reference_compaction: Option<ReferenceCompaction>,
}

struct ReferenceCompaction {
    states: Vec<StateManifest>,
    spans: Vec<Vec<ChunkSpan>>,
}

/// Outcome of staging one chunk: an already-durable dedup winner, or a freshly
/// encoded object queued for the next batched put.
enum StagedChunk {
    Deduplicated(ChunkRef),
    Encoded(ChunkObject),
}

impl ArtifactWriter {
    pub fn begin(
        store: Arc<LocalStore>,
        mut declaration: ManifestDeclaration,
        policy: WritePolicy,
    ) -> Result<Self, StoreError> {
        if !policy
            .qualified_semantic_models
            .contains(&declaration.semantic_model)
        {
            return Err(StoreError::Expectation(
                "semantic model is not in the production qualification set",
            ));
        }
        validate_family(&declaration.family)?;
        let family_id = representation_family_id(&declaration.family)?;
        if !policy
            .qualified_representation_families
            .contains(&family_id)
        {
            return Err(StoreError::Expectation(
                "representation family is not in the production qualification set",
            ));
        }
        if declaration.input_tokens.is_empty() {
            return Err(StoreError::State(
                "zero is a recomputation cut, not a durable artifact",
            ));
        }
        if declaration.states.len() != declaration.family.states.len() {
            return Err(StoreError::State(
                "writer declaration does not cover the complete family inventory",
            ));
        }
        for (declaration_state, family_state) in
            declaration.states.iter().zip(&declaration.family.states)
        {
            if declaration_state.key != family_state.key {
                return Err(StoreError::State(
                    "writer declarations do not match canonical family state order",
                ));
            }
        }
        let keys = store.schedule(store.key_epoch())?;
        let (input_cut, prefix_nodes) = derive_input_cut(
            keys.prefix_key(),
            &store.tenant_namespace(),
            &declaration.semantic_model,
            &declaration.family,
            &declaration.input_tokens,
            &declaration.auxiliary_inputs,
        )?;
        let exact_prefix_node = prefix_nodes.last().copied().ok_or(StoreError::State(
            "durable nonzero cut has no final prefix node",
        ))?;

        let mut reference_compaction = None;
        declaration.kind = match declaration.kind.clone() {
            ManifestKind::Full => ManifestKind::Full,
            ManifestKind::Delta {
                parent, parent_cut, ..
            } => {
                let parent_count = usize::try_from(parent_cut.token_count)
                    .map_err(|_| StoreError::State("delta parent cut exceeds usize"))?;
                let parent_tokens =
                    declaration
                        .input_tokens
                        .get(..parent_count)
                        .ok_or(StoreError::Expectation(
                            "delta parent cut is not a strict prefix of the child input",
                        ))?;
                let (derived_parent_cut, _) = derive_input_cut(
                    keys.prefix_key(),
                    &store.tenant_namespace(),
                    &declaration.semantic_model,
                    &declaration.family,
                    parent_tokens,
                    &declaration.auxiliary_inputs,
                )?;
                if derived_parent_cut != parent_cut {
                    return Err(StoreError::Expectation(
                        "delta parent cut does not identify the exact child prefix",
                    ));
                }

                let parent_bytes = store
                    .read_authenticated_manifest_object(&parent, &policy.validation_context())?;
                let parent_header = inspect_pack_header(&parent_bytes)?;
                let parent_keys = store.schedule(parent_header.key_epoch)?;
                let parent_manifest = decode_authenticated_pack(
                    &parent_bytes,
                    &parent_keys,
                    &policy.validation_context(),
                )?;
                if parent_manifest.input_cut != parent_cut
                    || parent_manifest.semantic_model != declaration.semantic_model
                    || parent_manifest.family != declaration.family
                    || parent_manifest.key_epoch != store.key_epoch()
                    || parent_manifest
                        .states
                        .iter()
                        .flat_map(|state| &state.chunks)
                        .any(|chunk| chunk.key_epoch != store.key_epoch())
                {
                    return Err(StoreError::Expectation(
                        "delta parent is not compatible with the declared child",
                    ));
                }
                let depth = parent_manifest
                    .realized_schema
                    .kind
                    .depth()
                    .checked_add(1)
                    .ok_or(StoreError::State("delta parent depth overflow"))?;
                if depth > MAX_DELTA_DEPTH {
                    let chain = authenticated_chain_for_compaction(
                        &store,
                        parent,
                        store.key_epoch(),
                        &declaration.semantic_model,
                        &declaration.family,
                        parent_cut,
                    )?;
                    reference_compaction = Some(reference_compaction_base(&chain, &declaration)?);
                }
                ManifestKind::Delta {
                    parent,
                    parent_cut,
                    depth,
                }
            }
        };
        validate_declaration_ranges(&declaration)?;
        let complete_restored = complete_restored_bytes(&declaration)?;
        if complete_restored > policy.maximum_restored_bytes {
            return Err(StoreError::Quota("artifact exceeds writer bound"));
        }
        let retained_references = reference_compaction
            .as_ref()
            .map(|compaction| {
                compaction
                    .states
                    .iter()
                    .map(|state| state.chunks.len() as u64)
                    .sum()
            })
            .unwrap_or(0);
        let expected_bytes = estimate_reservation(&declaration, retained_references)?;
        let intent_digest = artifact_intent_digest(&store, &declaration, input_cut, &policy)?;
        let state = store.reserve_upload(
            &policy.idempotency_key,
            UploadReservation {
                expected_bytes,
                publication_generation: policy.publication_generation,
                intent_digest,
                retention: policy.retention.with_physical_bytes(expected_bytes)?,
            },
        )?;
        let replayed = if state == UploadState::Published {
            let published = store.published_upload(&policy.idempotency_key)?.ok_or(
                StoreError::Authentication("published upload has no manifest catalog row"),
            )?;
            let bytes = store.read_authenticated_manifest_object(
                &published.manifest_id,
                &policy.validation_context(),
            )?;
            let existing_header = inspect_pack_header(&bytes)?;
            let existing_keys = store.schedule(existing_header.key_epoch)?;
            let existing =
                decode_authenticated_pack(&bytes, &existing_keys, &policy.validation_context())?;
            if !existing_matches_declaration(
                &existing,
                &declaration,
                input_cut,
                reference_compaction.as_ref(),
            )? {
                return Err(StoreError::Expectation(
                    "idempotency key was reused for a different artifact declaration",
                ));
            }
            Some(published)
        } else {
            store.mark_receiving(&policy.idempotency_key)?;
            None
        };
        Ok(Self {
            store,
            declaration,
            input_cut,
            exact_prefix_node,
            policy,
            keys,
            completed: Vec::new(),
            next_state: 0,
            poisoned: false,
            committed: replayed.is_some(),
            replayed,
            reference_compaction,
        })
    }

    pub fn published_replay(&self) -> Option<PublishedArtifact> {
        self.replayed
    }

    pub fn next_state(
        &mut self,
        expected_state_key: StateKey,
    ) -> Result<StateWriter<'_>, StoreError> {
        if self.replayed.is_some() {
            return Err(StoreError::State(
                "idempotency retry is already published; commit without writing states",
            ));
        }
        if self.poisoned {
            return Err(StoreError::Poisoned("artifact writer is poisoned"));
        }
        let declaration = self
            .declaration
            .states
            .get(self.next_state)
            .ok_or(StoreError::State(
                "all declared states were already written",
            ))?
            .clone();
        if declaration.key != expected_state_key {
            self.poisoned = true;
            return Err(StoreError::State(
                "next_state key does not match canonical declaration order",
            ));
        }
        let bytes_per_token = bytes_per_token(&self.declaration, self.next_state)?;
        let chunk_tokens = (MAX_CHUNK_PLAINTEXT as u64 / bytes_per_token).max(1);
        let chunk_capacity = usize::try_from(chunk_tokens * bytes_per_token)
            .map_err(|_| StoreError::State("chunk capacity exceeds usize"))?;
        Ok(StateWriter {
            writer: self,
            declaration,
            buffer: Vec::with_capacity(chunk_capacity),
            chunks: Vec::new(),
            pending: Vec::new(),
            pending_bytes: 0,
            total: 0,
            bytes_per_token,
            chunk_capacity,
            finished: false,
        })
    }

    fn store_chunk(
        &mut self,
        declaration: &StateDeclaration,
        plaintext: &[u8],
        span: ChunkSpan,
    ) -> Result<StagedChunk, StoreError> {
        let content_id = chunk_id(
            self.keys.chunk_identity_key(),
            &self.store.tenant_namespace(),
            &self.declaration.family,
            &declaration.key,
            &span,
            plaintext,
        )?;
        if let Some(existing) = self.store.find_chunk(&content_id, self.store.key_epoch())? {
            if existing.plaintext_bytes as usize != plaintext.len() {
                return Err(StoreError::Authentication(
                    "deduplicated chunk plaintext length mismatch",
                ));
            }
            return Ok(StagedChunk::Deduplicated(existing));
        }
        let stats_sidecar = self.stats_sidecar(declaration, plaintext, span)?;
        let object = encode_chunk_with_content_id(
            plaintext,
            &ChunkEncoding {
                tenant_namespace: self.store.tenant_namespace(),
                family: &self.declaration.family,
                state_key: &declaration.key,
                span,
                key_epoch: self.store.key_epoch(),
                encrypt: self.policy.encrypt_chunks,
                stats_sidecar: stats_sidecar.as_ref(),
            },
            &self.keys,
            Some(&content_id),
        )?;
        Ok(StagedChunk::Encoded(object))
    }

    /// Derive the optional M7 sidecar for one chunk from its fp16 state
    /// bytes.  The channel axis is the family state's `elements_per_token`;
    /// anything else fails the write closed rather than silently skipping.
    fn stats_sidecar(
        &self,
        declaration: &StateDeclaration,
        plaintext: &[u8],
        span: ChunkSpan,
    ) -> Result<Option<kvpack_core::StatsSidecar>, StoreError> {
        let Some(sink_count) = self.policy.stats_sidecar_sink_count else {
            return Ok(None);
        };
        let family_state = self
            .declaration
            .family
            .states
            .iter()
            .find(|state| state.key == declaration.key)
            .ok_or(StoreError::State(
                "written state is absent from the representation family",
            ))?;
        if family_state.dtype != kvpack_core::DType::F16 {
            return Err(StoreError::Expectation(
                "stats sidecars require an fp16 state dtype",
            ));
        }
        let channels = usize::try_from(family_state.elements_per_token)
            .map_err(|_| StoreError::State("state elements-per-token exceeds usize"))?;
        let tokens = usize::try_from(span.token_count)
            .map_err(|_| StoreError::State("chunk span token count exceeds usize"))?;
        let sidecar = kvpack_core::StatsSidecar::derive_f16(
            tokens,
            channels,
            sink_count as usize,
            plaintext,
        )?;
        Ok(Some(sidecar))
    }

    pub fn commit(mut self) -> Result<PublishedArtifact, StoreError> {
        if let Some(published) = self.replayed {
            return Ok(published);
        }
        if self.poisoned {
            return Err(StoreError::Poisoned("artifact writer is poisoned"));
        }
        if self.next_state != self.declaration.states.len() {
            self.poisoned = true;
            return Err(StoreError::State("not every declared state was written"));
        }
        let realized_schema = derive_realized_schema(&self.declaration, &self.completed)?;
        let completed = std::mem::take(&mut self.completed);
        let (realized_schema, states) = match self.reference_compaction.take() {
            Some(compaction) => compact_references(realized_schema, completed, compaction)?,
            None => (realized_schema, completed),
        };
        let manifest = CutManifest {
            tenant_namespace: self.store.tenant_namespace(),
            key_epoch: self.store.key_epoch(),
            semantic_model: self.declaration.semantic_model,
            input_cut: self.input_cut,
            family: self.declaration.family.clone(),
            realized_schema,
            states,
        };
        let encoded = encode_authenticated_pack(
            &manifest,
            &self.keys,
            self.policy.encrypt_manifest,
            &self.policy.validation_context(),
        )?;
        let published = self.store.publish_manifest(
            &self.policy.idempotency_key,
            &encoded,
            &manifest,
            std::slice::from_ref(&self.exact_prefix_node),
        )?;
        self.committed = true;
        Ok(published)
    }
}

impl Drop for ArtifactWriter {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.store.abort_upload(&self.policy.idempotency_key);
        }
    }
}

pub struct StateWriter<'a> {
    writer: &'a mut ArtifactWriter,
    declaration: StateDeclaration,
    buffer: Vec<u8>,
    chunks: Vec<Option<ChunkRef>>,
    pending: Vec<(usize, ChunkObject)>,
    pending_bytes: u64,
    total: u64,
    bytes_per_token: u64,
    chunk_capacity: usize,
    finished: bool,
}

impl StateWriter<'_> {
    pub fn write_all(&mut self, mut bytes: &[u8]) -> Result<(), StoreError> {
        if self.finished || self.writer.poisoned {
            return Err(StoreError::Poisoned("state writer is not writable"));
        }
        let required = segment_state_bytes(&self.writer.declaration, self.writer.next_state)?;
        if self
            .total
            .saturating_add(self.buffer.len() as u64)
            .saturating_add(bytes.len() as u64)
            > required
        {
            self.writer.poisoned = true;
            return Err(StoreError::State(
                "state writer received more bytes than its exact segment shape",
            ));
        }
        while !bytes.is_empty() {
            let take = (self.chunk_capacity - self.buffer.len()).min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buffer.len() == self.chunk_capacity {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), StoreError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if self.buffer.len() as u64 % self.bytes_per_token != 0 {
            self.writer.poisoned = true;
            return Err(StoreError::State(
                "chunk boundary splits one logical state token",
            ));
        }
        let plaintext = std::mem::take(&mut self.buffer);
        self.buffer = Vec::with_capacity(self.chunk_capacity);
        let token_offset = self.total / self.bytes_per_token;
        let base_plaintext_offset = self
            .declaration
            .logical_start
            .checked_mul(self.bytes_per_token)
            .ok_or(StoreError::State("state plaintext base offset overflow"))?;
        let span = ChunkSpan {
            token_start: self
                .declaration
                .logical_start
                .checked_add(token_offset)
                .ok_or(StoreError::State("chunk token start overflow"))?,
            token_count: plaintext.len() as u64 / self.bytes_per_token,
            plaintext_offset: base_plaintext_offset
                .checked_add(self.total)
                .ok_or(StoreError::State("chunk plaintext offset overflow"))?,
            plaintext_bytes: u32::try_from(plaintext.len())
                .map_err(|_| StoreError::State("chunk plaintext exceeds u32"))?,
        };
        match self.writer.store_chunk(&self.declaration, &plaintext, span) {
            Ok(StagedChunk::Deduplicated(reference)) => {
                self.total = self
                    .total
                    .checked_add(plaintext.len() as u64)
                    .ok_or(StoreError::State("state size overflow"))?;
                self.chunks.push(Some(reference));
                Ok(())
            }
            Ok(StagedChunk::Encoded(object)) => {
                self.total = self
                    .total
                    .checked_add(plaintext.len() as u64)
                    .ok_or(StoreError::State("state size overflow"))?;
                // Queue for the batched put: one directory sync set and one
                // catalog transaction per CHUNK_PUT_BATCH_BYTES (and always at
                // state end) instead of a 4-6 fsync storm per chunk.
                self.pending_bytes = self
                    .pending_bytes
                    .checked_add(object.bytes.len() as u64)
                    .ok_or(StoreError::State("pending chunk byte total overflow"))?;
                self.pending.push((self.chunks.len(), object));
                self.chunks.push(None);
                if self.pending_bytes >= CHUNK_PUT_BATCH_BYTES {
                    self.flush_pending()?;
                }
                Ok(())
            }
            Err(error) => {
                self.writer.poisoned = true;
                Err(error)
            }
        }
    }

    fn flush_pending(&mut self) -> Result<(), StoreError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let pending = std::mem::take(&mut self.pending);
        self.pending_bytes = 0;
        let objects: Vec<&ChunkObject> = pending.iter().map(|(_, object)| object).collect();
        let references = match self.writer.store.put_chunks_batch_with_retention(
            &objects,
            self.writer.store.key_epoch(),
            self.writer.policy.retention,
            false,
        ) {
            Ok(references) => references,
            Err(error) => {
                self.writer.poisoned = true;
                return Err(error);
            }
        };
        for ((slot, object), reference) in pending.iter().zip(references) {
            if reference.plaintext_bytes != object.plaintext_bytes {
                self.writer.poisoned = true;
                return Err(StoreError::Authentication(
                    "deduplicated chunk plaintext length mismatch",
                ));
            }
            self.chunks[*slot] = Some(reference);
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), StoreError> {
        self.flush()?;
        self.flush_pending()?;
        let required = segment_state_bytes(&self.writer.declaration, self.writer.next_state)?;
        if self.total != required {
            self.writer.poisoned = true;
            return Err(StoreError::State(
                "state writer ended before its exact segment shape was complete",
            ));
        }
        let chunks = std::mem::take(&mut self.chunks)
            .into_iter()
            .map(|slot| slot.ok_or(StoreError::State("state chunk reference was not resolved")))
            .collect::<Result<Vec<_>, _>>()?;
        self.writer.completed.push(StateManifest {
            key: self.declaration.key.clone(),
            chunks,
        });
        self.writer.next_state += 1;
        self.finished = true;
        Ok(())
    }
}

impl Drop for StateWriter<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.writer.poisoned = true;
        }
    }
}

mod helpers;
use helpers::{
    artifact_intent_digest, bytes_per_token, compact_references, complete_restored_bytes,
    derive_realized_schema, estimate_reservation, existing_matches_declaration,
    reference_compaction_base, segment_state_bytes, validate_declaration_ranges,
};
