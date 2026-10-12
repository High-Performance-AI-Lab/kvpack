use super::*;

pub(super) fn export_intent_digest(
    store: &LocalStore,
    declaration: &ExportDeclaration,
    final_cut: InputCutId,
    policy: &WritePolicy,
) -> Result<Id32, StoreError> {
    let mut digest = IntentHasher::new(b"kvpack/catalog/export-intent/v1");
    digest.id(&store.tenant_namespace());
    digest.u64(store.key_epoch());
    digest.id(&semantic_model_id(&declaration.semantic_model));
    digest.id(&representation_family_id(&declaration.family)?);
    digest.id(&final_cut.token_root);
    digest.id(&final_cut.auxiliary_input_root);
    digest.u64(final_cut.token_count);
    digest.u64(PREFIX_BLOCK_TOKENS as u64);
    digest.u64(declaration.states.len() as u64);
    for state in &declaration.states {
        digest.u32(state.key.layer);
        digest.bytes(state.key.state_name.as_bytes());
        digest.u64(state.strides.len() as u64);
        for stride in &state.strides {
            digest.u64(*stride);
        }
        digest.u32(state.atomic_group);
    }
    digest.byte(u8::from(policy.encrypt_chunks));
    digest.byte(u8::from(policy.encrypt_manifest));
    digest.u64(policy.maximum_restored_bytes);
    Ok(digest.finish())
}

pub(super) fn family_bytes_per_token(state: &FamilyState) -> Result<u64, StoreError> {
    state
        .elements_per_token
        .checked_mul(
            state
                .dtype
                .width_bytes()
                .ok_or(StoreError::State("family dtype has no fixed width"))?,
        )
        .ok_or(StoreError::State("state bytes-per-token overflow"))
}

pub(super) fn shape_for_cut(state: &FamilyState, token_count: u64) -> Result<Shape, StoreError> {
    let dimensions: Vec<_> = state
        .dimensions
        .iter()
        .map(|dimension| match dimension {
            kvpack_core::StaticDimension::Token => token_count,
            kvpack_core::StaticDimension::Fixed(value) => *value,
        })
        .collect();
    Ok(Shape::new(&dimensions)?)
}

fn contiguous_strides(shape: &Shape) -> Result<Vec<u64>, StoreError> {
    let mut strides = vec![0; shape.rank()];
    let mut stride = 1u64;
    for index in (0..shape.rank()).rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(shape.dims()[index])
            .ok_or(StoreError::State("contiguous export stride overflow"))?;
    }
    Ok(strides)
}

pub(super) fn strides_for_cut(
    declaration: &ExportStateDeclaration,
    family: &FamilyState,
    shape: &Shape,
) -> Result<Vec<u64>, StoreError> {
    if family.layout == Layout::Contiguous {
        contiguous_strides(shape)
    } else {
        Ok(declaration.strides.clone())
    }
}

pub(super) fn physical_footprint(
    shape: &Shape,
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
                .ok_or(StoreError::State("export physical footprint overflow"))
        })?;
    last.checked_add(1)
        .and_then(|elements| elements.checked_mul(width))
        .ok_or(StoreError::State("export physical footprint overflow"))
}

pub(super) fn validate_export_declaration(
    declaration: &ExportDeclaration,
    policy: &WritePolicy,
) -> Result<(), StoreError> {
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
            "zero is a recomputation cut, not a durable export",
        ));
    }
    if declaration.states.len() != declaration.family.states.len() {
        return Err(StoreError::State(
            "export declaration does not cover the complete family inventory",
        ));
    }
    let token_count = u64::try_from(declaration.input_tokens.len())
        .map_err(|_| StoreError::State("export token count exceeds u64"))?;
    let mut restored = 0u64;
    let mut physical = 0u64;
    for (state, family) in declaration.states.iter().zip(&declaration.family.states) {
        if state.key != family.key
            || state.atomic_group == 0
            || state.strides.len() != family.dimensions.len()
            || state.strides.contains(&0)
        {
            return Err(StoreError::State(
                "export state order, strides, or atomic group is invalid",
            ));
        }
        let shape = shape_for_cut(family, token_count)?;
        if family.layout == Layout::Contiguous && state.strides != contiguous_strides(&shape)? {
            return Err(StoreError::State(
                "contiguous export state has noncanonical final strides",
            ));
        }
        let bytes_per_token = family_bytes_per_token(family)?;
        if bytes_per_token == 0 || bytes_per_token > MAX_CHUNK_PLAINTEXT as u64 {
            return Err(StoreError::State(
                "one logical export token does not fit in a bounded chunk",
            ));
        }
        restored = restored
            .checked_add(
                token_count
                    .checked_mul(bytes_per_token)
                    .ok_or(StoreError::State("export restored bytes overflow"))?,
            )
            .ok_or(StoreError::State("export restored byte total overflow"))?;
        physical = physical
            .checked_add(physical_footprint(
                &shape,
                &state.strides,
                family
                    .dtype
                    .width_bytes()
                    .ok_or(StoreError::State("family dtype has no fixed width"))?,
            )?)
            .ok_or(StoreError::State("export physical byte total overflow"))?;
    }
    if restored > policy.maximum_restored_bytes || physical > policy.maximum_restored_bytes {
        return Err(StoreError::Quota("export exceeds writer resource bound"));
    }
    Ok(())
}

pub(super) fn next_chunk_bytes(
    token_start: u64,
    final_tokens: u64,
    bytes_per_token: u64,
    maximum_chunk_tokens: u64,
) -> Result<usize, StoreError> {
    if token_start >= final_tokens {
        return Ok(0);
    }
    let checkpoint = PREFIX_BLOCK_TOKENS as u64;
    let next_checkpoint = token_start
        .checked_div(checkpoint)
        .and_then(|block| block.checked_add(1))
        .and_then(|block| block.checked_mul(checkpoint))
        .ok_or(StoreError::State("export checkpoint boundary overflow"))?;
    let tokens = maximum_chunk_tokens
        .min(next_checkpoint - token_start)
        .min(final_tokens - token_start);
    usize::try_from(
        tokens
            .checked_mul(bytes_per_token)
            .ok_or(StoreError::State("export chunk capacity overflow"))?,
    )
    .map_err(|_| StoreError::State("export chunk capacity exceeds usize"))
}

fn chunks_at_cut(cut: u64, maximum_chunk_tokens: u64) -> Result<u64, StoreError> {
    let checkpoint = PREFIX_BLOCK_TOKENS as u64;
    let full = cut / checkpoint;
    let remainder = cut % checkpoint;
    full.checked_mul(checkpoint.div_ceil(maximum_chunk_tokens))
        .and_then(|value| value.checked_add(remainder.div_ceil(maximum_chunk_tokens)))
        .ok_or(StoreError::State("export chunk count overflow"))
}

pub(super) fn estimate_export_reservation(
    declaration: &ExportDeclaration,
    prefix_nodes: &[PrefixNode],
) -> Result<u64, StoreError> {
    let final_tokens = u64::try_from(declaration.input_tokens.len())
        .map_err(|_| StoreError::State("export token count exceeds u64"))?;
    let mut plaintext = 0u64;
    let mut final_chunks = 0u64;
    let mut manifest_estimate = 0u64;
    for family in &declaration.family.states {
        let per_token = family_bytes_per_token(family)?;
        plaintext = plaintext
            .checked_add(
                final_tokens
                    .checked_mul(per_token)
                    .ok_or(StoreError::State("export reservation overflow"))?,
            )
            .ok_or(StoreError::State("export reservation overflow"))?;
        let maximum = (MAX_CHUNK_PLAINTEXT as u64 / per_token).max(1);
        final_chunks = final_chunks
            .checked_add(chunks_at_cut(final_tokens, maximum)?)
            .ok_or(StoreError::State("export reservation overflow"))?;
    }
    for node in prefix_nodes {
        let chunks = declaration.family.states.iter().try_fold(
            0u64,
            |total, family| -> Result<u64, StoreError> {
                let per_token = family_bytes_per_token(family)?;
                let maximum = (MAX_CHUNK_PLAINTEXT as u64 / per_token).max(1);
                total
                    .checked_add(chunks_at_cut(node.token_count, maximum)?)
                    .ok_or(StoreError::State("export reservation overflow"))
            },
        )?;
        manifest_estimate = manifest_estimate
            .checked_add(8192)
            .and_then(|value| value.checked_add(declaration.states.len() as u64 * 2048))
            .and_then(|value| value.checked_add(chunks * 192))
            .ok_or(StoreError::State("export reservation overflow"))?;
    }
    plaintext
        .checked_add(final_chunks.saturating_mul(12 * 4096))
        .and_then(|value| value.checked_add(manifest_estimate))
        .ok_or(StoreError::State("export reservation overflow"))
}
