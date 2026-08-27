use std::collections::{BTreeMap, BTreeSet};

use crate::identity::validate_state_key;
use crate::manifest::is_zero_id;
use crate::{
    CutManifest, FamilyState, Layout, ManifestKind, PackError, RepresentationFamilyId,
    RepresentationMode, Shape, StateKey, StaticDimension, TokenAxisRule, ALIGNMENT,
    CHUNK_HEADER_BYTES, MAX_CHUNKS_PER_STATE, MAX_CHUNK_OBJECT_BYTES, MAX_CHUNK_PLAINTEXT,
    MAX_DELTA_DEPTH, MAX_STATES,
};

#[derive(Debug, Clone, Copy)]
pub struct ManifestBounds {
    pub max_states: usize,
    pub max_chunks_per_state: usize,
    pub max_restored_bytes: u64,
}

impl Default for ManifestBounds {
    fn default() -> Self {
        Self {
            max_states: MAX_STATES,
            max_chunks_per_state: MAX_CHUNKS_PER_STATE,
            max_restored_bytes: 4 * 1024 * 1024 * 1024 * 1024,
        }
    }
}

/// Immutable parser/resource bounds.  Feature qualification is not a request
/// switch: the v1 type system itself contains only accepted lanes.
#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    pub bounds: ManifestBounds,
}

fn checked_product(
    values: impl IntoIterator<Item = u64>,
    error: &'static str,
) -> Result<u64, PackError> {
    values.into_iter().try_fold(1u64, |product, value| {
        product.checked_mul(value).ok_or(PackError::Bounds(error))
    })
}

fn validate_dependencies(states: &[FamilyState]) -> Result<(), PackError> {
    let keys: BTreeSet<&StateKey> = states.iter().map(|state| &state.key).collect();
    let graph: BTreeMap<&StateKey, &[StateKey]> = states
        .iter()
        .map(|state| (&state.key, state.dependencies.as_slice()))
        .collect();
    for (key, dependencies) in &graph {
        let mut previous: Option<&StateKey> = None;
        for dependency in *dependencies {
            if dependency == *key {
                return Err(PackError::Graph("state depends on itself"));
            }
            if !keys.contains(dependency) {
                return Err(PackError::Graph("state dependency is missing"));
            }
            if previous.is_some_and(|value| value >= dependency) {
                return Err(PackError::Graph(
                    "state dependencies are not unique canonical order",
                ));
            }
            previous = Some(dependency);
        }
    }

    fn visit<'a>(
        key: &'a StateKey,
        graph: &BTreeMap<&'a StateKey, &'a [StateKey]>,
        visiting: &mut BTreeSet<&'a StateKey>,
        done: &mut BTreeSet<&'a StateKey>,
    ) -> Result<(), PackError> {
        if done.contains(key) {
            return Ok(());
        }
        if !visiting.insert(key) {
            return Err(PackError::Graph("state dependency cycle"));
        }
        if let Some(dependencies) = graph.get(key) {
            for dependency in *dependencies {
                visit(dependency, graph, visiting, done)?;
            }
        }
        visiting.remove(key);
        done.insert(key);
        Ok(())
    }

    let mut visiting = BTreeSet::new();
    let mut done = BTreeSet::new();
    for key in graph.keys() {
        visit(key, &graph, &mut visiting, &mut done)?;
    }
    Ok(())
}

pub fn validate_family(family: &RepresentationFamilyId) -> Result<(), PackError> {
    if is_zero_id(&family.engine_cache_abi)
        || is_zero_id(&family.topology)
        || is_zero_id(&family.shard_map)
    {
        return Err(PackError::Semantics(
            "representation family contains a zero identity",
        ));
    }
    if family.page_size_tokens == 0 {
        return Err(PackError::Semantics(
            "representation family page size must be nonzero",
        ));
    }
    if family.states.is_empty() || family.states.len() > MAX_STATES {
        return Err(PackError::Bounds("family state count is outside bounds"));
    }
    let mut previous: Option<&StateKey> = None;
    for state in &family.states {
        validate_state_key(&state.key)?;
        if previous.is_some_and(|key| key >= &state.key) {
            return Err(PackError::Semantics(
                "family states are not unique canonical order",
            ));
        }
        previous = Some(&state.key);
        if state.codec_version != 1 {
            return Err(PackError::Codec("unsupported family codec version"));
        }
        let rank = state.dimensions.len();
        if rank == 0 || rank > crate::MAX_RANK || state.token_axis as usize >= rank {
            return Err(PackError::Semantics(
                "family token axis is outside its rank",
            ));
        }
        let token_dimensions = state
            .dimensions
            .iter()
            .enumerate()
            .filter(|(_, dimension)| matches!(dimension, StaticDimension::Token))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if token_dimensions.as_slice() != [state.token_axis as usize] {
            return Err(PackError::Semantics(
                "family must have exactly one declared token dimension",
            ));
        }
        let elements_per_token = checked_product(
            state
                .dimensions
                .iter()
                .filter_map(|dimension| match dimension {
                    StaticDimension::Token => None,
                    StaticDimension::Fixed(value) => Some(*value),
                }),
            "family elements-per-token overflows u64",
        )?;
        if elements_per_token != state.elements_per_token || elements_per_token == 0 {
            return Err(PackError::Semantics(
                "family elements-per-token is not canonical",
            ));
        }
        if state.dtype.width_bytes().is_none() {
            return Err(PackError::Semantics("family dtype has no fixed width"));
        }
        if state.layout == Layout::Contiguous && state.token_axis_rule == TokenAxisRule::Gather {
            return Err(PackError::Semantics(
                "contiguous family cannot require strided token gathering",
            ));
        }
        if family.mode == RepresentationMode::Portable
            && (state.layout != Layout::Contiguous
                || !matches!(
                    state.token_axis_rule,
                    TokenAxisRule::Direct | TokenAxisRule::TailWindow
                ))
        {
            return Err(PackError::Semantics(
                "portable family requires contiguous direct token streams",
            ));
        }
    }
    validate_dependencies(&family.states)
}

fn validate_shape(
    shape: &Shape,
    family: &FamilyState,
    expected_tokens: u64,
) -> Result<(), PackError> {
    if shape.rank() != family.dimensions.len() {
        return Err(PackError::Semantics(
            "realized shape rank does not match family rank",
        ));
    }
    for (index, (actual, dimension)) in shape.dims().iter().zip(&family.dimensions).enumerate() {
        let expected = match dimension {
            StaticDimension::Token => expected_tokens,
            StaticDimension::Fixed(value) => *value,
        };
        if *actual != expected
            || (index == family.token_axis as usize) != matches!(dimension, StaticDimension::Token)
        {
            return Err(PackError::Semantics(
                "realized shape does not satisfy static family dimensions",
            ));
        }
    }
    Ok(())
}

fn physical_footprint(shape: &Shape, strides: &[u64], width: u64) -> Result<u64, PackError> {
    let last_element =
        shape
            .dims()
            .iter()
            .zip(strides)
            .try_fold(0u64, |offset, (dimension, stride)| {
                dimension
                    .checked_sub(1)
                    .and_then(|count| count.checked_mul(*stride))
                    .and_then(|term| offset.checked_add(term))
                    .ok_or(PackError::Bounds("state physical footprint overflows u64"))
            })?;
    last_element
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(width))
        .ok_or(PackError::Bounds("state physical footprint overflows u64"))
}

fn validate_atomic_groups(manifest: &CutManifest) -> Result<(), PackError> {
    let expected: BTreeSet<_> = manifest
        .family
        .states
        .iter()
        .map(|state| state.key.clone())
        .collect();
    let mut observed = BTreeSet::new();
    let mut previous_id = 0u32;
    for group in &manifest.realized_schema.atomic_groups {
        if group.id == 0 || group.id <= previous_id {
            return Err(PackError::Semantics(
                "atomic groups are not unique canonical order",
            ));
        }
        previous_id = group.id;
        let mut previous_key: Option<&StateKey> = None;
        for state in &group.states {
            if !expected.contains(state) {
                return Err(PackError::Graph("atomic group names an unknown state"));
            }
            if previous_key.is_some_and(|key| key >= state) {
                return Err(PackError::Semantics(
                    "atomic-group states are not unique canonical order",
                ));
            }
            if !observed.insert(state.clone()) {
                return Err(PackError::Graph("state appears in multiple atomic groups"));
            }
            previous_key = Some(state);
        }
    }
    if observed != expected {
        return Err(PackError::Graph(
            "atomic groups do not cover the complete family inventory",
        ));
    }
    Ok(())
}

pub fn validate_manifest(
    manifest: &CutManifest,
    context: &ValidationContext,
) -> Result<(), PackError> {
    if is_zero_id(&manifest.tenant_namespace) {
        return Err(PackError::Semantics("tenant namespace is zero"));
    }
    if manifest.key_epoch == 0 {
        return Err(PackError::Semantics("key epoch must be nonzero"));
    }
    let semantic_ids = [
        manifest.semantic_model.weights_config,
        manifest.semantic_model.adapters,
        manifest.semantic_model.tokenizer_template,
        manifest.semantic_model.position_semantics,
        manifest.semantic_model.qualified_math,
    ];
    if semantic_ids.iter().any(is_zero_id) {
        return Err(PackError::Semantics(
            "semantic model identity contains a zero component",
        ));
    }
    if is_zero_id(&manifest.input_cut.token_root)
        || is_zero_id(&manifest.input_cut.auxiliary_input_root)
        || manifest.input_cut.token_count == 0
    {
        return Err(PackError::Semantics("input cut identity is invalid"));
    }
    validate_family(&manifest.family)?;

    let (cut_start, cut_count) = match &manifest.realized_schema.kind {
        ManifestKind::Full => (0, manifest.input_cut.token_count),
        ManifestKind::Delta {
            parent,
            parent_cut,
            depth,
        } => {
            if is_zero_id(parent)
                || is_zero_id(&parent_cut.token_root)
                || is_zero_id(&parent_cut.auxiliary_input_root)
            {
                return Err(PackError::Graph("delta parent identity is invalid"));
            }
            if *depth == 0 || *depth > MAX_DELTA_DEPTH {
                return Err(PackError::Graph("delta depth is outside 1..=7"));
            }
            if parent_cut.auxiliary_input_root != manifest.input_cut.auxiliary_input_root
                || parent_cut.token_count == 0
                || parent_cut.token_count >= manifest.input_cut.token_count
            {
                return Err(PackError::Graph(
                    "delta parent cut is not a compatible strict prefix",
                ));
            }
            (
                parent_cut.token_count,
                manifest.input_cut.token_count - parent_cut.token_count,
            )
        }
    };

    let schema = &manifest.realized_schema;
    if schema.states.len() != manifest.family.states.len()
        || manifest.states.len() != manifest.family.states.len()
        || schema.states.len() > context.bounds.max_states
    {
        return Err(PackError::Bounds(
            "manifest, family, and schema state counts differ or exceed bounds",
        ));
    }
    validate_atomic_groups(manifest)?;

    let mut segment_total = 0u64;
    let mut complete_total = 0u64;
    for ((family_state, realized), payload) in manifest
        .family
        .states
        .iter()
        .zip(&schema.states)
        .zip(&manifest.states)
    {
        if family_state.key != realized.key || realized.key != payload.key {
            return Err(PackError::Semantics(
                "family, realized schema, and payload state order differ",
            ));
        }
        let (range_start, range_count) =
            if family_state.token_axis_rule == TokenAxisRule::TailWindow {
                if !matches!(schema.kind, ManifestKind::Full)
                    || realized.logical_count == 0
                    || realized.logical_count > manifest.input_cut.token_count
                {
                    return Err(PackError::Semantics(
                        "tail-window states require one nonempty full snapshot",
                    ));
                }
                (
                    manifest.input_cut.token_count - realized.logical_count,
                    realized.logical_count,
                )
            } else {
                (cut_start, cut_count)
            };
        let full_tokens = if family_state.token_axis_rule == TokenAxisRule::TailWindow {
            range_count
        } else {
            manifest.input_cut.token_count
        };
        validate_shape(&realized.full_shape, family_state, full_tokens)?;
        validate_shape(&realized.segment_shape, family_state, range_count)?;
        if realized.logical_start != range_start
            || realized.logical_count != range_count
            || realized.absolute_position != manifest.input_cut.token_count
            || realized.window != 0
        {
            return Err(PackError::Semantics(
                "realized state range does not match the exact manifest cut",
            ));
        }
        if realized.strides.len() != realized.full_shape.rank() || realized.strides.contains(&0) {
            return Err(PackError::Semantics(
                "realized state strides do not match its rank",
            ));
        }
        if family_state.layout == Layout::Contiguous {
            let mut expected_stride = 1u64;
            for (dimension, stride) in realized
                .full_shape
                .dims()
                .iter()
                .zip(&realized.strides)
                .rev()
            {
                if *stride != expected_stride {
                    return Err(PackError::Semantics(
                        "contiguous state has noncanonical strides",
                    ));
                }
                expected_stride =
                    expected_stride
                        .checked_mul(*dimension)
                        .ok_or(PackError::Bounds(
                            "contiguous stride calculation overflows u64",
                        ))?;
            }
        }
        let width = family_state
            .dtype
            .width_bytes()
            .ok_or(PackError::Semantics("family dtype has no fixed width"))?;
        let segment_bytes = realized
            .segment_shape
            .element_count()?
            .checked_mul(width)
            .ok_or(PackError::Bounds("segment byte count overflows u64"))?;
        let complete_bytes = realized
            .full_shape
            .element_count()?
            .checked_mul(width)
            .ok_or(PackError::Bounds("complete byte count overflows u64"))?;
        let physical_token_offset = if family_state.token_axis_rule == TokenAxisRule::TailWindow {
            0
        } else {
            range_start
        };
        let physical_offset = physical_token_offset
            .checked_mul(realized.strides[family_state.token_axis as usize])
            .and_then(|value| value.checked_mul(width))
            .ok_or(PackError::Bounds("physical token offset overflows u64"))?;
        let physical_span = physical_footprint(&realized.segment_shape, &realized.strides, width)?;
        let complete_physical = physical_footprint(&realized.full_shape, &realized.strides, width)?;
        if realized.physical_offset_bytes != physical_offset
            || realized.physical_span_bytes != physical_span
            || realized.complete_physical_bytes != complete_physical
        {
            return Err(PackError::Semantics(
                "realized physical span is not canonical",
            ));
        }
        if realized.chunk_spans.is_empty()
            || realized.chunk_spans.len() > context.bounds.max_chunks_per_state
            || payload.chunks.len() != realized.chunk_spans.len()
        {
            return Err(PackError::Bounds("state chunk count is outside bounds"));
        }
        let bytes_per_token = family_state
            .elements_per_token
            .checked_mul(width)
            .ok_or(PackError::Bounds("bytes per token overflows u64"))?;
        if bytes_per_token == 0 || bytes_per_token > MAX_CHUNK_PLAINTEXT as u64 {
            return Err(PackError::Bounds(
                "one state token does not fit in a bounded chunk",
            ));
        }
        let mut expected_token = range_start;
        let mut expected_offset =
            range_start
                .checked_mul(bytes_per_token)
                .ok_or(PackError::Bounds(
                    "state plaintext base offset overflows u64",
                ))?;
        for (span, chunk) in realized.chunk_spans.iter().zip(&payload.chunks) {
            if span.token_start != expected_token
                || span.token_count == 0
                || span.plaintext_offset != expected_offset
            {
                return Err(PackError::Semantics(
                    "chunk spans are not a contiguous token/byte partition",
                ));
            }
            let expected_bytes = span
                .token_count
                .checked_mul(bytes_per_token)
                .ok_or(PackError::Bounds("chunk span byte count overflows u64"))?;
            if expected_bytes != span.plaintext_bytes as u64
                || expected_bytes == 0
                || expected_bytes > MAX_CHUNK_PLAINTEXT as u64
                || chunk.plaintext_bytes != span.plaintext_bytes
            {
                return Err(PackError::Semantics(
                    "chunk span has a noncanonical plaintext size",
                ));
            }
            if is_zero_id(&chunk.chunk_id)
                || is_zero_id(&chunk.object_key)
                || is_zero_id(&chunk.object_digest)
                || chunk.key_epoch == 0
            {
                return Err(PackError::Semantics(
                    "chunk reference contains a zero identity",
                ));
            }
            if chunk.object_bytes as usize % ALIGNMENT != 0
                || chunk.object_bytes as usize <= CHUNK_HEADER_BYTES
                || chunk.object_bytes as usize > MAX_CHUNK_OBJECT_BYTES
            {
                return Err(PackError::Semantics("chunk object size is not aligned"));
            }
            expected_token = expected_token
                .checked_add(span.token_count)
                .ok_or(PackError::Bounds("chunk token range overflows u64"))?;
            expected_offset = expected_offset
                .checked_add(expected_bytes)
                .ok_or(PackError::Bounds("chunk byte range overflows u64"))?;
        }
        let range_end = range_start
            .checked_add(range_count)
            .ok_or(PackError::Bounds("state token range overflows u64"))?;
        let expected_end_offset =
            range_end
                .checked_mul(bytes_per_token)
                .ok_or(PackError::Bounds(
                    "state plaintext end offset overflows u64",
                ))?;
        if expected_token != range_end || expected_offset != expected_end_offset {
            return Err(PackError::Semantics(
                "chunk spans do not cover the complete realized state segment",
            ));
        }
        segment_total = segment_total
            .checked_add(segment_bytes)
            .ok_or(PackError::Bounds(
                "segment restored byte total overflows u64",
            ))?;
        complete_total = complete_total
            .checked_add(complete_bytes)
            .ok_or(PackError::Bounds(
                "complete restored byte total overflows u64",
            ))?;
    }
    if schema.segment_restored_bytes != segment_total
        || schema.complete_restored_bytes != complete_total
        || schema.complete_restored_bytes > context.bounds.max_restored_bytes
    {
        return Err(PackError::Bounds(
            "manifest restored byte totals are invalid",
        ));
    }
    Ok(())
}
