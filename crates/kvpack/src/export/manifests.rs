use std::collections::BTreeMap;

use kvpack_core::{
    AtomicGroup, CutManifest, InputCutId, ManifestKind, PrefixNode, RealizedCutSchemaId,
    RealizedStateSchema, StateKey, StateManifest,
};

use super::{
    family_bytes_per_token, physical_footprint, shape_for_cut, strides_for_cut, ExportSession,
    ExportStateDeclaration, StoredState,
};
use crate::{LocalStore, StoreError};
use kvpack_core::{Id32, RepresentationFamilyId, SemanticModelId};

impl ExportSession {
    pub(super) fn manifest_for_cut(
        &self,
        node: PrefixNode,
        kind: ManifestKind,
    ) -> Result<CutManifest, StoreError> {
        manifest_for_cut_parts(
            ManifestParts {
                store: &self.store,
                semantic_model: self.semantic_model,
                family: &self.family,
                declarations: &self.declarations,
                completed: &self.completed,
                auxiliary_root: self.auxiliary_root,
            },
            node,
            kind,
        )
    }
}

pub(super) struct ManifestParts<'a> {
    pub store: &'a LocalStore,
    pub semantic_model: SemanticModelId,
    pub family: &'a RepresentationFamilyId,
    pub declarations: &'a [ExportStateDeclaration],
    pub completed: &'a [StoredState],
    pub auxiliary_root: Id32,
}

pub(super) fn manifest_for_cut_parts(
    parts: ManifestParts<'_>,
    node: PrefixNode,
    kind: ManifestKind,
) -> Result<CutManifest, StoreError> {
    let ManifestParts {
        store,
        semantic_model,
        family,
        declarations,
        completed,
        auxiliary_root,
    } = parts;
    let cut = node.token_count;
    let logical_start = match &kind {
        ManifestKind::Full => 0,
        ManifestKind::Delta { parent_cut, .. } => parent_cut.token_count,
    };
    let logical_count = cut
        .checked_sub(logical_start)
        .ok_or(StoreError::State("export manifest range is reversed"))?;
    if logical_count == 0 {
        return Err(StoreError::State("export manifest range is empty"));
    }
    let mut states = Vec::with_capacity(completed.len());
    let mut schemas = Vec::with_capacity(completed.len());
    let mut groups: BTreeMap<u32, Vec<StateKey>> = BTreeMap::new();
    let mut segment_restored_bytes = 0u64;
    let mut complete_restored_bytes = 0u64;
    for (index, stored) in completed.iter().enumerate() {
        let family_state = &family.states[index];
        let declaration = &declarations[index];
        if stored.key != family_state.key || stored.key != declaration.key {
            return Err(StoreError::State(
                "completed state inventory changed before publication",
            ));
        }
        let selected: Vec<_> = stored
            .chunks
            .iter()
            .skip_while(|chunk| {
                chunk
                    .span
                    .token_start
                    .checked_add(chunk.span.token_count)
                    .is_some_and(|end| end <= logical_start)
            })
            .take_while(|chunk| {
                chunk
                    .span
                    .token_start
                    .checked_add(chunk.span.token_count)
                    .is_some_and(|end| end <= cut)
            })
            .cloned()
            .collect();
        let selected_start = selected.first().map(|chunk| chunk.span.token_start);
        let selected_end = selected
            .last()
            .and_then(|chunk| chunk.span.token_start.checked_add(chunk.span.token_count));
        if selected_start != Some(logical_start) || selected_end != Some(cut) {
            return Err(StoreError::State(
                "cut-aware chunk inventory does not cover the exact manifest range",
            ));
        }
        let full_shape = shape_for_cut(family_state, cut)?;
        let segment_shape = shape_for_cut(family_state, logical_count)?;
        let strides = strides_for_cut(declaration, family_state, &full_shape)?;
        let width = family_state
            .dtype
            .width_bytes()
            .ok_or(StoreError::State("family dtype has no fixed width"))?;
        let physical_offset_bytes = logical_start
            .checked_mul(strides[family_state.token_axis as usize])
            .and_then(|value| value.checked_mul(width))
            .ok_or(StoreError::State("cut physical offset overflow"))?;
        let physical_span_bytes = physical_footprint(&segment_shape, &strides, width)?;
        let complete_physical_bytes = physical_footprint(&full_shape, &strides, width)?;
        let bytes_per_token = family_bytes_per_token(family_state)?;
        let segment_bytes = logical_count
            .checked_mul(bytes_per_token)
            .ok_or(StoreError::State("cut restored bytes overflow"))?;
        let complete_bytes = cut
            .checked_mul(bytes_per_token)
            .ok_or(StoreError::State("complete cut restored bytes overflow"))?;
        segment_restored_bytes = segment_restored_bytes
            .checked_add(segment_bytes)
            .ok_or(StoreError::State("cut restored byte total overflow"))?;
        complete_restored_bytes =
            complete_restored_bytes
                .checked_add(complete_bytes)
                .ok_or(StoreError::State(
                    "complete cut restored byte total overflow",
                ))?;
        groups
            .entry(declaration.atomic_group)
            .or_default()
            .push(stored.key.clone());
        states.push(StateManifest {
            key: stored.key.clone(),
            chunks: selected
                .iter()
                .map(|chunk| chunk.reference.clone())
                .collect(),
        });
        schemas.push(RealizedStateSchema {
            key: stored.key.clone(),
            full_shape,
            segment_shape,
            strides,
            logical_start,
            logical_count,
            physical_offset_bytes,
            physical_span_bytes,
            complete_physical_bytes,
            absolute_position: cut,
            window: 0,
            chunk_spans: selected.into_iter().map(|chunk| chunk.span).collect(),
        });
    }
    Ok(CutManifest {
        tenant_namespace: store.tenant_namespace(),
        key_epoch: store.key_epoch(),
        semantic_model,
        input_cut: InputCutId {
            token_root: node.id,
            auxiliary_input_root: auxiliary_root,
            token_count: cut,
        },
        family: family.clone(),
        realized_schema: RealizedCutSchemaId {
            kind,
            states: schemas,
            atomic_groups: groups
                .into_iter()
                .map(|(id, states)| AtomicGroup { id, states })
                .collect(),
            segment_restored_bytes,
            complete_restored_bytes,
        },
        states,
    })
}
