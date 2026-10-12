use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kvpack_core::{
    decode_authenticated_pack, decode_chunk, inspect_pack_header, CutManifest, Id32, InputCutId,
    ManifestKind, RealizedCutSchemaId, RepresentationFamilyId, SemanticModelId, StateDeclaration,
    StateKey, ValidationContext,
};

use crate::store::{family_digest, semantic_digest};
use crate::{LocalStore, RestoreStatePlan, StoreError, VerifiedRestoreSink};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactLocator {
    Manifest(Id32),
    Prefix {
        nodes: Vec<kvpack_core::PrefixNode>,
        candidate_bound: usize,
    },
}

#[derive(Debug, Clone)]
pub struct OpenExpectations {
    pub locator: ArtifactLocator,
    pub semantic_model: SemanticModelId,
    pub input_cut: InputCutId,
    pub family: RepresentationFamilyId,
    pub realized_schema: RealizedCutSchemaId,
    pub minimum_key_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreSelection {
    states: BTreeSet<StateKey>,
}

impl RestoreSelection {
    pub fn new(states: impl IntoIterator<Item = StateKey>) -> Result<Self, StoreError> {
        let mut result = BTreeSet::new();
        for state in states {
            if !result.insert(state) {
                return Err(StoreError::Expectation(
                    "restore selection contains a duplicate state",
                ));
            }
        }
        if result.is_empty() {
            return Err(StoreError::Expectation("restore selection is empty"));
        }
        Ok(Self { states: result })
    }

    pub fn all(manifest: &CutManifest) -> Self {
        Self {
            states: manifest
                .states
                .iter()
                .map(|state| state.key.clone())
                .collect(),
        }
    }
}

pub struct AuthenticatedArtifact {
    store: Arc<LocalStore>,
    manifest_id: Id32,
    manifest: CutManifest,
}

impl AuthenticatedArtifact {
    pub fn open(
        store: Arc<LocalStore>,
        expectations: OpenExpectations,
    ) -> Result<Self, StoreError> {
        let manifest_id = match &expectations.locator {
            ArtifactLocator::Manifest(id) => *id,
            ArtifactLocator::Prefix {
                nodes,
                candidate_bound,
            } => {
                let semantic = semantic_digest(&expectations.semantic_model);
                let family = family_digest(&expectations.family)?;
                store
                    .resolve_prefix(nodes, &semantic, &family, *candidate_bound)?
                    .ok_or(StoreError::NotFound)?
                    .manifest_id
            }
        };
        let bytes = store.read_manifest_bytes(&manifest_id)?;
        let header = inspect_pack_header(&bytes)?;
        if header.tenant_namespace != store.tenant_namespace() {
            return Err(StoreError::Authentication(
                "manifest belongs to another tenant",
            ));
        }
        let minimum_key_epoch = expectations
            .minimum_key_epoch
            .max(store.minimum_readable_key_epoch());
        if header.key_epoch < minimum_key_epoch || header.key_epoch > store.key_epoch() {
            return Err(StoreError::NotFound);
        }
        let keys = store.schedule(header.key_epoch)?;
        let context = ValidationContext::default();
        let manifest = decode_authenticated_pack(&bytes, &keys, &context)?;
        if manifest
            .states
            .iter()
            .flat_map(|state| &state.chunks)
            .any(|chunk| chunk.key_epoch < minimum_key_epoch || chunk.key_epoch > store.key_epoch())
        {
            return Err(StoreError::NotFound);
        }
        if kvpack_core::manifest_id(&manifest.encode_canonical()?) != manifest_id {
            return Err(StoreError::Authentication(
                "catalog manifest identity mismatch",
            ));
        }
        if manifest.semantic_model != expectations.semantic_model {
            return Err(StoreError::Expectation("semantic model identity mismatch"));
        }
        if manifest.input_cut != expectations.input_cut {
            return Err(StoreError::Expectation("input cut identity mismatch"));
        }
        if manifest.family != expectations.family {
            return Err(StoreError::Expectation(
                "representation family identity mismatch",
            ));
        }
        if manifest.realized_schema != expectations.realized_schema {
            return Err(StoreError::Expectation(
                "realized cut schema identity mismatch",
            ));
        }
        Ok(Self {
            store,
            manifest_id,
            manifest,
        })
    }

    pub fn manifest(&self) -> &CutManifest {
        &self.manifest
    }

    pub fn manifest_id(&self) -> Id32 {
        self.manifest_id
    }

    pub fn restore_selected(
        &self,
        selection: &RestoreSelection,
        sink: &mut dyn VerifiedRestoreSink,
    ) -> Result<(), StoreError> {
        if !matches!(self.manifest.realized_schema.kind, ManifestKind::Full) {
            return Err(StoreError::State(
                "delta manifests require an authenticated parent-chain restore plan",
            ));
        }
        let index_by_key: BTreeMap<_, _> = self
            .manifest
            .states
            .iter()
            .enumerate()
            .map(|(index, state)| (state.key.clone(), index))
            .collect();
        for key in &selection.states {
            if !index_by_key.contains_key(key) {
                return Err(StoreError::Expectation(
                    "selected state is not in the authenticated manifest",
                ));
            }
        }
        let mut group_by_key = BTreeMap::new();
        for group in &self.manifest.realized_schema.atomic_groups {
            let group_states: BTreeSet<_> = group.states.iter().cloned().collect();
            if group_states
                .iter()
                .any(|key| selection.states.contains(key))
                && !group_states.is_subset(&selection.states)
            {
                return Err(StoreError::Expectation(
                    "restore selection splits an authenticated atomic group",
                ));
            }
            for key in &group.states {
                group_by_key.insert(key.clone(), group.id);
            }
        }
        let selected_indices: Vec<_> = self
            .manifest
            .states
            .iter()
            .enumerate()
            .filter(|(_, state)| selection.states.contains(&state.key))
            .map(|(index, _)| index)
            .collect();
        let plans: Vec<_> = selected_indices
            .iter()
            .map(|index| {
                let schema = &self.manifest.realized_schema.states[*index];
                let declaration = StateDeclaration {
                    key: schema.key.clone(),
                    full_shape: schema.full_shape,
                    segment_shape: schema.segment_shape,
                    strides: schema.strides.clone(),
                    logical_start: schema.logical_start,
                    logical_count: schema.logical_count,
                    absolute_position: schema.absolute_position,
                    window: schema.window,
                    atomic_group: group_by_key[&schema.key],
                };
                RestoreStatePlan {
                    declaration,
                    plaintext_bytes: schema
                        .chunk_spans
                        .iter()
                        .map(|span| span.plaintext_bytes as u64)
                        .sum(),
                    physical_span_bytes: schema.complete_physical_bytes,
                    atomic_group: group_by_key[&schema.key],
                    chunk_count: schema.chunk_spans.len(),
                }
            })
            .collect();
        if let Err(error) = sink.begin_restore(self.manifest_id, &plans) {
            sink.abort_restore();
            return Err(error);
        }
        for index in selected_indices {
            let state = &self.manifest.states[index];
            let schema = &self.manifest.realized_schema.states[index];
            for (chunk, span) in state.chunks.iter().zip(&schema.chunk_spans) {
                let result = (|| {
                    let bytes = self.store.pin_chunk(chunk)?.read_all()?;
                    let keys = self.store.schedule(chunk.key_epoch)?;
                    let plaintext = decode_chunk(
                        &bytes,
                        chunk,
                        span,
                        &self.manifest.tenant_namespace,
                        &self.manifest.family,
                        &state.key,
                        &keys,
                    )?;
                    sink.write_verified_chunk(&state.key, span.plaintext_offset, &plaintext)
                })();
                if let Err(error) = result {
                    sink.abort_restore();
                    return Err(error);
                }
            }
        }
        if let Err(error) = sink.commit_restore() {
            sink.abort_restore();
            return Err(error);
        }
        Ok(())
    }

    pub fn scrub_full(&self) -> Result<(), StoreError> {
        struct ScrubSink;
        impl VerifiedRestoreSink for ScrubSink {
            fn begin_restore(&mut self, _: Id32, _: &[RestoreStatePlan]) -> Result<(), StoreError> {
                Ok(())
            }
            fn write_verified_chunk(
                &mut self,
                _: &StateKey,
                _: u64,
                _: &[u8],
            ) -> Result<(), StoreError> {
                Ok(())
            }
            fn commit_restore(&mut self) -> Result<(), StoreError> {
                Ok(())
            }
            fn abort_restore(&mut self) {}
        }
        self.restore_selected(&RestoreSelection::all(&self.manifest), &mut ScrubSink)
    }
}
