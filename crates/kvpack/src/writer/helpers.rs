use super::*;

pub(super) fn bytes_per_token(
    declaration: &ManifestDeclaration,
    state_index: usize,
) -> Result<u64, StoreError> {
    let family = declaration
        .family
        .states
        .get(state_index)
        .ok_or(StoreError::State("state index exceeds family"))?;
    family
        .elements_per_token
        .checked_mul(
            family
                .dtype
                .width_bytes()
                .ok_or(StoreError::State("family dtype has no fixed width"))?,
        )
        .ok_or(StoreError::State("state bytes-per-token overflow"))
}

pub(super) fn segment_state_bytes(
    declaration: &ManifestDeclaration,
    state_index: usize,
) -> Result<u64, StoreError> {
    let state = declaration
        .states
        .get(state_index)
        .ok_or(StoreError::State("state index exceeds declaration"))?;
    let family = &declaration.family.states[state_index];
    state
        .segment_shape
        .element_count()?
        .checked_mul(
            family
                .dtype
                .width_bytes()
                .ok_or(StoreError::State("family dtype has no fixed width"))?,
        )
        .ok_or(StoreError::State("segment state byte size overflow"))
}

fn complete_state_bytes(
    declaration: &ManifestDeclaration,
    state_index: usize,
) -> Result<u64, StoreError> {
    let state = &declaration.states[state_index];
    let family = &declaration.family.states[state_index];
    state
        .full_shape
        .element_count()?
        .checked_mul(
            family
                .dtype
                .width_bytes()
                .ok_or(StoreError::State("family dtype has no fixed width"))?,
        )
        .ok_or(StoreError::State("complete state byte size overflow"))
}

pub(super) fn complete_restored_bytes(
    declaration: &ManifestDeclaration,
) -> Result<u64, StoreError> {
    (0..declaration.states.len()).try_fold(0u64, |sum, index| {
        sum.checked_add(complete_state_bytes(declaration, index)?)
            .ok_or(StoreError::State("artifact restored size overflow"))
    })
}

pub(super) fn validate_declaration_ranges(
    declaration: &ManifestDeclaration,
) -> Result<(), StoreError> {
    let child_cut = u64::try_from(declaration.input_tokens.len())
        .map_err(|_| StoreError::State("input token count exceeds u64"))?;
    let (cut_start, cut_count) = match &declaration.kind {
        ManifestKind::Full => (0, child_cut),
        ManifestKind::Delta { parent_cut, .. } => {
            if parent_cut.token_count >= child_cut {
                return Err(StoreError::State(
                    "delta parent must be a strict input prefix",
                ));
            }
            (parent_cut.token_count, child_cut - parent_cut.token_count)
        }
    };
    for (state, family) in declaration.states.iter().zip(&declaration.family.states) {
        let (start, count) = if family.token_axis_rule == TokenAxisRule::TailWindow {
            if !matches!(declaration.kind, ManifestKind::Full)
                || state.logical_count == 0
                || state.logical_count > child_cut
            {
                return Err(StoreError::State(
                    "tail-window states require one nonempty full snapshot",
                ));
            }
            (child_cut - state.logical_count, state.logical_count)
        } else {
            (cut_start, cut_count)
        };
        let rank = family.dimensions.len();
        if state.atomic_group == 0
            || state.full_shape.rank() != rank
            || state.segment_shape.rank() != rank
            || state.strides.len() != rank
            || state.strides.contains(&0)
            || family.token_axis as usize >= rank
        {
            return Err(StoreError::State(
                "state declaration rank, strides, token axis, or atomic group is invalid",
            ));
        }
        for (index, dimension) in family.dimensions.iter().enumerate() {
            let expected_full = match dimension {
                StaticDimension::Token if family.token_axis_rule == TokenAxisRule::TailWindow => {
                    count
                }
                StaticDimension::Token => child_cut,
                StaticDimension::Fixed(value) => *value,
            };
            let expected_segment = match dimension {
                StaticDimension::Token => count,
                StaticDimension::Fixed(value) => *value,
            };
            if state.full_shape.dims()[index] != expected_full
                || state.segment_shape.dims()[index] != expected_segment
            {
                return Err(StoreError::State(
                    "state declaration shape does not satisfy its static family",
                ));
            }
        }
        if family.layout == kvpack_core::Layout::Contiguous {
            let mut expected_stride = 1u64;
            for (dimension, stride) in state.full_shape.dims().iter().zip(&state.strides).rev() {
                if *stride != expected_stride {
                    return Err(StoreError::State(
                        "contiguous state declaration has noncanonical strides",
                    ));
                }
                expected_stride = expected_stride
                    .checked_mul(*dimension)
                    .ok_or(StoreError::State("contiguous state stride overflow"))?;
            }
        }
        let per_token = family
            .elements_per_token
            .checked_mul(
                family
                    .dtype
                    .width_bytes()
                    .ok_or(StoreError::State("family dtype has no fixed width"))?,
            )
            .ok_or(StoreError::State("state bytes-per-token overflow"))?;
        if per_token == 0 || per_token > MAX_CHUNK_PLAINTEXT as u64 {
            return Err(StoreError::State(
                "one logical state token does not fit in a bounded chunk",
            ));
        }
        if state.logical_start != start
            || state.logical_count != count
            || state.absolute_position != child_cut
            || state.window != 0
        {
            return Err(StoreError::State(
                "state declaration does not match full/delta cut range",
            ));
        }
    }
    Ok(())
}

fn physical_footprint(
    shape: &kvpack_core::Shape,
    strides: &[u64],
    width: u64,
) -> Result<u64, StoreError> {
    let last = shape
        .dims()
        .iter()
        .zip(strides)
        .try_fold(0u64, |offset, (dimension, stride)| {
            dimension
                .checked_sub(1)
                .and_then(|count| count.checked_mul(*stride))
                .and_then(|term| offset.checked_add(term))
                .ok_or(StoreError::State("state physical footprint overflow"))
        })?;
    last.checked_add(1)
        .and_then(|elements| elements.checked_mul(width))
        .ok_or(StoreError::State("state physical footprint overflow"))
}

pub(super) fn derive_realized_schema(
    declaration: &ManifestDeclaration,
    payloads: &[StateManifest],
) -> Result<RealizedCutSchemaId, StoreError> {
    let mut states = Vec::with_capacity(declaration.states.len());
    let mut segment_total = 0u64;
    let mut complete_total = 0u64;
    let mut groups: BTreeMap<u32, Vec<StateKey>> = BTreeMap::new();
    for (index, (state, payload)) in declaration.states.iter().zip(payloads).enumerate() {
        let family = &declaration.family.states[index];
        let width = family
            .dtype
            .width_bytes()
            .ok_or(StoreError::State("family dtype has no fixed width"))?;
        let bytes_per_token = family
            .elements_per_token
            .checked_mul(width)
            .ok_or(StoreError::State("state bytes-per-token overflow"))?;
        let segment_bytes = segment_state_bytes(declaration, index)?;
        let complete_bytes = complete_state_bytes(declaration, index)?;
        let physical_token_offset = if family.token_axis_rule == TokenAxisRule::TailWindow {
            0
        } else {
            state.logical_start
        };
        let physical_offset_bytes = physical_token_offset
            .checked_mul(state.strides[family.token_axis as usize])
            .and_then(|value| value.checked_mul(width))
            .ok_or(StoreError::State("state physical offset overflow"))?;
        let mut plaintext_offset = state
            .logical_start
            .checked_mul(bytes_per_token)
            .ok_or(StoreError::State("state plaintext base offset overflow"))?;
        let mut token_start = state.logical_start;
        let mut chunk_spans = Vec::with_capacity(payload.chunks.len());
        for chunk in &payload.chunks {
            let token_count = chunk.plaintext_bytes as u64 / bytes_per_token;
            chunk_spans.push(ChunkSpan {
                token_start,
                token_count,
                plaintext_offset,
                plaintext_bytes: chunk.plaintext_bytes,
            });
            token_start = token_start
                .checked_add(token_count)
                .ok_or(StoreError::State("chunk token range overflow"))?;
            plaintext_offset = plaintext_offset
                .checked_add(chunk.plaintext_bytes as u64)
                .ok_or(StoreError::State("chunk plaintext range overflow"))?;
        }
        states.push(RealizedStateSchema {
            key: state.key.clone(),
            full_shape: state.full_shape,
            segment_shape: state.segment_shape,
            strides: state.strides.clone(),
            logical_start: state.logical_start,
            logical_count: state.logical_count,
            physical_offset_bytes,
            physical_span_bytes: physical_footprint(&state.segment_shape, &state.strides, width)?,
            complete_physical_bytes: physical_footprint(&state.full_shape, &state.strides, width)?,
            absolute_position: state.absolute_position,
            window: state.window,
            chunk_spans,
        });
        groups
            .entry(state.atomic_group)
            .or_default()
            .push(state.key.clone());
        segment_total = segment_total
            .checked_add(segment_bytes)
            .ok_or(StoreError::State("segment restored byte total overflow"))?;
        complete_total = complete_total
            .checked_add(complete_bytes)
            .ok_or(StoreError::State("complete restored byte total overflow"))?;
    }
    let atomic_groups = groups
        .into_iter()
        .map(|(id, states)| AtomicGroup { id, states })
        .collect();
    Ok(RealizedCutSchemaId {
        kind: declaration.kind.clone(),
        states,
        atomic_groups,
        segment_restored_bytes: segment_total,
        complete_restored_bytes: complete_total,
    })
}

fn declarations_from_manifest(manifest: &CutManifest) -> Result<Vec<StateDeclaration>, StoreError> {
    let mut groups = BTreeMap::new();
    for group in &manifest.realized_schema.atomic_groups {
        for state in &group.states {
            groups.insert(state.clone(), group.id);
        }
    }
    manifest
        .realized_schema
        .states
        .iter()
        .map(|state| {
            Ok(StateDeclaration {
                key: state.key.clone(),
                full_shape: state.full_shape,
                segment_shape: state.segment_shape,
                strides: state.strides.clone(),
                logical_start: state.logical_start,
                logical_count: state.logical_count,
                absolute_position: state.absolute_position,
                window: state.window,
                atomic_group: *groups.get(&state.key).ok_or(StoreError::Authentication(
                    "authenticated schema has no atomic group",
                ))?,
            })
        })
        .collect()
}

pub(super) fn reference_compaction_base(
    chain: &[CutManifest],
    declaration: &ManifestDeclaration,
) -> Result<ReferenceCompaction, StoreError> {
    if chain.len() != MAX_DELTA_DEPTH as usize + 1
        || chain
            .last()
            .is_none_or(|manifest| manifest.realized_schema.kind.depth() != MAX_DELTA_DEPTH)
    {
        return Err(StoreError::Authentication(
            "reference compaction requires one full base plus seven deltas",
        ));
    }
    let mut states: Vec<_> = declaration
        .states
        .iter()
        .map(|state| StateManifest {
            key: state.key.clone(),
            chunks: Vec::new(),
        })
        .collect();
    let mut spans = vec![Vec::new(); declaration.states.len()];
    for manifest in chain {
        if manifest.states.len() != states.len()
            || manifest.realized_schema.states.len() != states.len()
        {
            return Err(StoreError::Authentication(
                "reference compaction chain state inventory changed",
            ));
        }
        for (index, (payload, schema)) in manifest
            .states
            .iter()
            .zip(&manifest.realized_schema.states)
            .enumerate()
        {
            if payload.key != states[index].key
                || schema.key != states[index].key
                || payload.chunks.len() != schema.chunk_spans.len()
            {
                return Err(StoreError::Authentication(
                    "reference compaction chain payload and schema disagree",
                ));
            }
            states[index].chunks.extend(payload.chunks.iter().cloned());
            spans[index].extend(schema.chunk_spans.iter().copied());
        }
    }
    let parent_cut = chain.last().unwrap().input_cut.token_count;
    for index in 0..states.len() {
        let per_token = bytes_per_token(declaration, index)?;
        let expected_end = parent_cut
            .checked_mul(per_token)
            .ok_or(StoreError::State("compaction parent byte range overflow"))?;
        let actual_end = spans[index].last().and_then(|span| {
            span.plaintext_offset
                .checked_add(span.plaintext_bytes as u64)
        });
        let chunk_tokens = (MAX_CHUNK_PLAINTEXT as u64 / per_token).max(1);
        let appended_chunks = declaration.states[index]
            .logical_count
            .div_ceil(chunk_tokens);
        if actual_end != Some(expected_end)
            || states[index].chunks.len() as u64 + appended_chunks > MAX_CHUNKS_PER_STATE as u64
        {
            return Err(StoreError::Quota(
                "reference compaction exceeds its authenticated chunk bound",
            ));
        }
    }
    Ok(ReferenceCompaction { states, spans })
}

pub(super) fn compact_references(
    mut appended_schema: RealizedCutSchemaId,
    appended_states: Vec<StateManifest>,
    mut compaction: ReferenceCompaction,
) -> Result<(RealizedCutSchemaId, Vec<StateManifest>), StoreError> {
    if appended_states.len() != compaction.states.len()
        || appended_schema.states.len() != compaction.states.len()
    {
        return Err(StoreError::State(
            "reference compaction state inventory changed during append",
        ));
    }
    for (index, appended) in appended_states.into_iter().enumerate() {
        if appended.key != compaction.states[index].key
            || appended_schema.states[index].key != appended.key
        {
            return Err(StoreError::State(
                "reference compaction state order changed during append",
            ));
        }
        compaction.states[index].chunks.extend(appended.chunks);
        compaction.spans[index].append(&mut appended_schema.states[index].chunk_spans);
        if compaction.states[index].chunks.len() != compaction.spans[index].len()
            || compaction.states[index].chunks.len() > MAX_CHUNKS_PER_STATE
        {
            return Err(StoreError::Quota(
                "reference compaction exceeds its authenticated chunk bound",
            ));
        }
        let realized = &mut appended_schema.states[index];
        realized.segment_shape = realized.full_shape;
        realized.logical_start = 0;
        realized.logical_count = realized.absolute_position;
        realized.physical_offset_bytes = 0;
        realized.physical_span_bytes = realized.complete_physical_bytes;
        realized.chunk_spans = std::mem::take(&mut compaction.spans[index]);
    }
    appended_schema.kind = ManifestKind::Full;
    appended_schema.segment_restored_bytes = appended_schema.complete_restored_bytes;
    Ok((appended_schema, compaction.states))
}

pub(super) fn existing_matches_declaration(
    existing: &CutManifest,
    declaration: &ManifestDeclaration,
    input_cut: InputCutId,
    compaction: Option<&ReferenceCompaction>,
) -> Result<bool, StoreError> {
    if existing.semantic_model != declaration.semantic_model
        || existing.input_cut != input_cut
        || existing.family != declaration.family
    {
        return Ok(false);
    }
    let existing_declarations = declarations_from_manifest(existing)?;
    let Some(compaction) = compaction else {
        return Ok(existing.realized_schema.kind == declaration.kind
            && existing_declarations == declaration.states);
    };
    if !matches!(existing.realized_schema.kind, ManifestKind::Full)
        || existing.states.len() != compaction.states.len()
        || existing.realized_schema.states.len() != compaction.spans.len()
    {
        return Ok(false);
    }
    let mut expected = declaration.states.clone();
    for state in &mut expected {
        state.segment_shape = state.full_shape;
        state.logical_start = 0;
        state.logical_count = input_cut.token_count;
    }
    if existing_declarations != expected {
        return Ok(false);
    }
    Ok(existing
        .states
        .iter()
        .zip(&existing.realized_schema.states)
        .zip(compaction.states.iter().zip(&compaction.spans))
        .all(|((payload, schema), (base_payload, base_spans))| {
            payload.chunks.starts_with(&base_payload.chunks)
                && schema.chunk_spans.starts_with(base_spans)
        }))
}

pub(super) fn artifact_intent_digest(
    store: &LocalStore,
    declaration: &ManifestDeclaration,
    input_cut: InputCutId,
    policy: &WritePolicy,
) -> Result<Id32, StoreError> {
    let mut digest = IntentHasher::new(b"kvpack/catalog/artifact-intent/v1");
    digest.id(&store.tenant_namespace());
    digest.u64(store.key_epoch());
    digest.id(&semantic_model_id(&declaration.semantic_model));
    digest.id(&representation_family_id(&declaration.family)?);
    hash_input_cut(&mut digest, input_cut);
    match &declaration.kind {
        ManifestKind::Full => digest.byte(0),
        ManifestKind::Delta {
            parent,
            parent_cut,
            depth,
        } => {
            digest.byte(1);
            digest.id(parent);
            hash_input_cut(&mut digest, *parent_cut);
            digest.byte(*depth);
        }
    }
    digest.u64(declaration.states.len() as u64);
    for state in &declaration.states {
        digest.bytes(&state.canonical_schema_bytes()?);
    }
    digest.byte(u8::from(policy.encrypt_chunks));
    digest.byte(u8::from(policy.encrypt_manifest));
    digest.u64(policy.maximum_restored_bytes);
    Ok(digest.finish())
}

fn hash_input_cut(digest: &mut IntentHasher, cut: InputCutId) {
    digest.id(&cut.token_root);
    digest.id(&cut.auxiliary_input_root);
    digest.u64(cut.token_count);
}

pub(super) fn estimate_reservation(
    declaration: &ManifestDeclaration,
    retained_references: u64,
) -> Result<u64, StoreError> {
    let segment = (0..declaration.states.len()).try_fold(0u64, |sum, index| {
        sum.checked_add(segment_state_bytes(declaration, index)?)
            .ok_or(StoreError::State("segment restored size overflow"))
    })?;
    let chunks = declaration
        .states
        .iter()
        .enumerate()
        .try_fold(0u64, |sum, (index, state)| {
            let per_token = bytes_per_token(declaration, index)?;
            let chunk_tokens = (MAX_CHUNK_PLAINTEXT as u64 / per_token).max(1);
            sum.checked_add(state.logical_count.div_ceil(chunk_tokens))
                .ok_or(StoreError::State("chunk count overflow"))
        })?;
    segment
        .checked_add(chunks.saturating_mul(12 * 4096))
        .and_then(|value| {
            value.checked_add(
                8192 + declaration.states.len() as u64 * 2048
                    + chunks.saturating_add(retained_references) * 192,
            )
        })
        .ok_or(StoreError::State("write reservation overflow"))
}
